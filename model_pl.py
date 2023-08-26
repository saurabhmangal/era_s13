"""
Implementation of YOLOv3 architecture using pytorch lightning
"""

import lightning.pytorch as pl
import torch.optim as optim
import torch.nn as nn
import torch
from loss import YoloLoss
from torch.optim.lr_scheduler import OneCycleLR
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples,
    non_max_suppression,plot_image
)
import config
import time

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
model_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3Lightning(pl.LightningModule):
    
    def __init__(self,num_classes,in_channels=3,learning_rate = 2e-4,len_train_loader=10):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.len_train_loader = len_train_loader
        self.layers = self._create_conv_layers()
        self.loss_fn = YoloLoss()
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(config.DEVICE)
        self.threshold=config.CONF_THRESHOLD
        self.train_state = []
        self.test_state = []
        self.counter = 1

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue
            x = layer(x)
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in model_config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers
    
    def training_step(self, batch, batch_idx):
        stage ="train"
        x,y = batch
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0
        y0, y1, y2 = (
            y[0],
            y[1],
            y[2]
        )

        # with torch.cuda.amp.autocast():
        out = self(x)
        loss = (
            self.loss_fn(out[0], y0, self.scaled_anchors[0])
            + self.loss_fn(out[1], y1, self.scaled_anchors[1])
            + self.loss_fn(out[2], y2, self.scaled_anchors[2])
        )
        
        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=True, logger=True)
        
        for i in range(3):
            obj = y[i][..., 0] == 1
            noobj = y[i][..., 0] == 0
            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > self.threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)
        
        self.log(f"Class {stage} Accuracy",(correct_class/(tot_class_preds+1e-16))*100)
        self.log(f"No Obj {stage} Accuracy",(correct_class/(tot_class_preds+1e-16))*100)
        self.log(f"Obj {stage} Accuracy",(correct_class/(tot_class_preds+1e-16))*100)
        # print(f"Class {stage} accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
        # print(f"No obj {stage} accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
        # print(f"Obj {stage} accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
        self.log(f"{stage} loss",loss.item())
        self.train_state.append({f"{stage} loss":loss.item(),
                                 f"Obj {stage} accuracy": f"{(correct_obj/(tot_obj+1e-16))*100:2f}%",
                                 f"No obj {stage} accuracy is": f"{(correct_noobj/(tot_noobj+1e-16))*100:2f}%",
                                 f"Class {stage} accuracy is": f"{(correct_class/(tot_class_preds+1e-16))*100:2f}%"
                                 })

        
        
        thresh = 0.6
        iou_thresh = config.NMS_IOU_THRESH
        if batch_idx%517==0 and batch_idx!=0:
            bboxes = [[] for _ in range(x.shape[0])]
            for i in range(3):
                batch_size, A, S, _, _ = out[i].shape
                anchor = self.scaled_anchors[i]
                boxes_scale_i = cells_to_bboxes(
                    out[i], anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box
            for i in range(batch_size//4):
                nms_boxes = non_max_suppression(
                    bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
                )
                myfig = plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, self.trainer.current_epoch )
                # self.log({"training_inference_image":[wandb.Image(myfig, caption="training_inference_image")]})
                self.counter+=1
        return loss
    
    def on_train_epoch_end(self):
        print("Epoch:",self.trainer.current_epoch," ->",self.train_state[-1])
        # plot_couple_examples(self, test_loader, 0.6, 0.5, self.scaled_anchors)
        
    def on_test_epoch_end(self):
        print("Epoch:",self.trainer.current_epoch," ->",self.test_state[-1])
    
    def on_validation_epoch_end(self):
        print("Epoch:",self.trainer.current_epoch," ->",self.test_state[-1])
        

    def evaluate(self, batch,batch_idx, stage=None):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0
        x, y = batch
        out = self(x)
        loss = (
            self.loss_fn(out[0], y[0], self.scaled_anchors[0])
            + self.loss_fn(out[1], y[1], self.scaled_anchors[1])
            + self.loss_fn(out[2], y[2], self.scaled_anchors[2])
        )
        for i in range(3):
            obj = y[i][..., 0] == 1
            noobj = y[i][..., 0] == 0
            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > self.threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)
        
        self.log(f"Class {stage} Accuracy",(correct_class/(tot_class_preds+1e-16))*100)
        self.log(f"No Obj {stage} Accuracy",(correct_class/(tot_class_preds+1e-16))*100)
        self.log(f"Obj {stage} Accuracy",(correct_class/(tot_class_preds+1e-16))*100)
        # print(f"Class {stage} accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
        # print(f"No obj {stage} accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
        # print(f"Obj {stage} accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
        self.test_state.append({f"{stage} loss":loss.item(),
                                 f"Obj {stage} accuracy": f"{(correct_obj/(tot_obj+1e-16))*100:2f}%",
                                 f"No obj {stage} accuracy is": f"{(correct_noobj/(tot_noobj+1e-16))*100:2f}%",
                                 f"Class {stage} accuracy is": f"{(correct_class/(tot_class_preds+1e-16))*100:2f}%"
                                 })
        self.log(f"{stage} loss",loss.item())
        thresh = 0.6
        iou_thresh = config.NMS_IOU_THRESH
        if batch_idx%517==0 and batch_idx!=0:
            bboxes = [[] for _ in range(x.shape[0])]
            for i in range(3):
                batch_size, A, S, _, _ = out[i].shape
                anchor = self.scaled_anchors[i]
                boxes_scale_i = cells_to_bboxes(
                    out[i], anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box
            for i in range(batch_size//4):
                nms_boxes = non_max_suppression(
                    bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
                )
                myfig = plot_image(x[i].permute(1,2,0).detach().cpu(), nms_boxes, self.trainer.current_epoch )
                # self.log({"testing_inference_image":[wandb.Image(myfig, caption="testing_inference_image")]})
                
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx,"val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx,"test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                self.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
            )
        EPOCHS = config.NUM_EPOCHS * 2 // 5
        scheduler_dict = { "scheduler":torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1E-3,
            # steps_per_epoch=len(train_loader),
            steps_per_epoch=519,
            epochs=EPOCHS,
            pct_start=5/EPOCHS,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        ),"interval": "step"}

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3Lightning(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert out[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert out[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert out[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")




