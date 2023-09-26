
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
import copy
import logging
import itertools
import math
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList

############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        x = torch.flatten(x)
        #return F.log_softmax(x, dim = 1)
        return x

###############  Prototype Network  ##################
class Prtotype_Net(nn.Module):
    def __init__(self, output_shape = 128, ndf1=512):
        super(Prtotype_Net, self).__init__()

        self.linear1 = nn.Linear(1024, ndf1)
        self.linear2 = nn.Linear(ndf1, output_shape)
        #self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        #print("Protoproto")
        return x


################ Gradient reverse Layer  #######################
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)


@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        prototype_layer: int,
        use_ema: int,
        contra: int,
        num_classes: int
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.num_classes = num_classes

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"
        self.prototype_layer = prototype_layer
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.contra = contra
        self.dis_type = dis_type
        self.use_ema = use_ema
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type])  # Need to know the channel
        if (self.prototype_layer):
            self.proto = Prtotype_Net()

    def build_prototype(self):
        self.prototype_s1 = torch.zeros((self.num_classes, 128), device=torch.device('cuda'))
        self.prototype_s1 = self.prototype_s1
        self.number_of_occurance_s1 = [0] * self.num_classes
        self.prototype_s2 = torch.zeros((self.num_classes, 128), device=torch.device('cuda'))
        self.prototype_s2 = self.prototype_s2
        self.number_of_occurance_s2 = [0] * self.num_classes
        self.prototype_t = torch.zeros((self.num_classes, 128), device=torch.device('cuda'))
        self.number_of_occurance_t = [0] * self.num_classes
        self.prototype_t = self.prototype_t
        self.prototype_c = torch.zeros((self.num_classes, 128), device=torch.device('cuda'))
        self.number_of_occurance_c = [0] * self.num_classes
        self.prototype_c = self.prototype_c

    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            "prototype_layer": cfg.SEMISUPNET.PROTOTYPE_LAYER,
            "use_ema": cfg.SEMISUPNET.USE_EMA,
            "contra": cfg.SEMISUPNET.USE_CONTRA,
            "num_classes": len(cfg.DATALOADER.CLASS)
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def convert_gt_to_rcn(self, gt):
        temp = copy.deepcopy(gt)
        for item in temp:
            item.set('objectness_logits', torch.ones(len(item)).to(self.device))
            item.set('proposal_boxes', item.get('gt_boxes'))
            item.remove('gt_classes')
            item.remove('gt_boxes')
        return temp

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, target_type = 1
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = target_type
        target_label = 3

        if branch == "domain":
            images_s, images_t = self.preprocess_image_train(batched_inputs)
            features = self.backbone(images_s.tensor)
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            criterion = torch.nn.CrossEntropyLoss()
            loss_D_img_s = criterion(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))/D_img_out_s.shape[0]
            loss_D_img_t = 0
            if (target_type == 2):
                features_t = self.backbone(images_t.tensor)
                features_t = grad_reverse(features_t[self.dis_type])
                D_img_out_t = self.D_img(features_t)
                loss_D_img_t = criterion(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))/D_img_out_t.shape[0]

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s * .01
            losses["loss_D_img_t"] = loss_D_img_t * .01
            return losses, [], [], None

        elif(branch == "prototype_all"):
            self.build_prototype()
            target, source1, source2 = batched_inputs
            images_t = self.preprocess_image(target)
            images_s1 = self.preprocess_image(source1)
            images_s2 = self.preprocess_image(source2)

            gt_instances_t = [x["instances"].to(self.device) for x in target]
            gt_instances_s1 = [x["instances"].to(self.device) for x in source1]
            gt_instances_s2 = [x["instances"].to(self.device) for x in source2]

            features_t = self.backbone(images_t.tensor)
            features_s1 = self.backbone(images_s1.tensor)
            features_s2 = self.backbone(images_s2.tensor)

            proposals_rpn_t = self.convert_gt_to_rcn(gt_instances_t)
            proposals_rpn_s1 = self.convert_gt_to_rcn(gt_instances_s1)
            proposals_rpn_s2 = self.convert_gt_to_rcn(gt_instances_s2)

            gt_labels_t = gt_instances_t[0].gt_classes
            gt_labels_s1 = gt_instances_s1[0].gt_classes
            gt_labels_s2 = gt_instances_s2[0].gt_classes
            for i, x in enumerate(gt_instances_t):
                if (i != 0):
                    gt_labels_t = torch.cat((gt_labels_t, x.gt_classes))
            for i, x in enumerate(gt_instances_s1):
                if (i != 0):
                    gt_labels_s1 = torch.cat((gt_labels_s1, x.gt_classes))
            for i, x in enumerate(gt_instances_s2):
                if (i != 0):
                    gt_labels_s2 = torch.cat((gt_labels_s2, x.gt_classes))
            
            box_features_t = self.roi_heads(
                images_t,
                features_t,
                proposals_rpn_t,
                compute_loss=True,
                targets=gt_instances_t,
                branch=branch,
            )
            box_features_s1 = self.roi_heads(
                images_s1,
                features_s1,
                proposals_rpn_s1,
                compute_loss=True,
                targets=gt_instances_s1,
                branch=branch,
            )
            box_features_s2 = self.roi_heads(
                images_s2,
                features_s2,
                proposals_rpn_s2,
                compute_loss=True,
                targets=gt_instances_s2,
                branch=branch,
            )
            if (self.prototype_layer):
                box_features_t = self.proto(box_features_t)
                box_features_s1 = self.proto(box_features_s1)
                box_features_s2 = self.proto(box_features_s2)
                
            for lab, pro in zip(gt_labels_t, box_features_t):
                self.prototype_t[lab] = ((self.prototype_t[lab] * self.number_of_occurance_t[lab]) + pro)/(self.number_of_occurance_t[lab] + 1)
                self.number_of_occurance_t[lab] += 1

            for lab, pro in zip(gt_labels_s1, box_features_s1):
                self.prototype_s1[lab] = ((self.prototype_s1[lab] * self.number_of_occurance_s1[lab]) + pro)/(self.number_of_occurance_s1[lab] + 1)
                self.number_of_occurance_s1[lab] += 1
                if (self.contra):
                    self.prototype_c[lab] = ((self.prototype_c[lab] * self.number_of_occurance_c[lab]) + pro)/(self.number_of_occurance_c[lab] + 1)
                    self.number_of_occurance_c[lab] += 1

            for lab, pro in zip(gt_labels_s2, box_features_s2):
                self.prototype_s2[lab] = ((self.prototype_s2[lab] * self.number_of_occurance_s2[lab]) + pro)/(self.number_of_occurance_s2[lab] + 1)
                self.number_of_occurance_s2[lab] += 1
                if (self.contra):
                    self.prototype_c[lab] = ((self.prototype_c[lab] * self.number_of_occurance_c[lab]) + pro)/(self.number_of_occurance_c[lab] + 1)
                    self.number_of_occurance_c[lab] += 1
            
            loss_pro1 = F.cosine_similarity(self.prototype_s1, self.prototype_t).abs().mean().to(self.device)
            loss_pro2 = F.cosine_similarity(self.prototype_s2, self.prototype_t).abs().mean().to(self.device)
            loss_pro3 = F.cosine_similarity(self.prototype_s1, self.prototype_s2).abs().mean().to(self.device)
            loss = -(loss_pro1 + loss_pro2 + loss_pro3)/3
            self.prototype_t.detach_()
            self.prototype_s1.detach_()
            self.prototype_s2.detach_()

            loss_c = 0
            if self.contra:
                for i, j in itertools.combinations(list(range(len(self.prototype_s1))),2):
                    loss_c += F.cosine_similarity(self.prototype_s1[i].view(-1,1), self.prototype_s1[j].view(-1,1)).abs().mean().to(self.device)
                loss_c = loss_c/math.comb(len(self.prototype_s1), 2)
                self.prototype_s1.detach_()

            #print(f"Pro loss  :   {loss}")
            return loss, loss_c

        elif(branch == "prototype_all2"):
            source1, source2 = batched_inputs
            images_s1 = self.preprocess_image(source1)
            images_s2 = self.preprocess_image(source2)

            gt_instances_s1 = [x["instances"].to(self.device) for x in source1]
            gt_instances_s2 = [x["instances"].to(self.device) for x in source2]

            features_s1 = self.backbone(images_s1.tensor)
            features_s2 = self.backbone(images_s2.tensor)
            
            proposals_rpn_s1 = self.convert_gt_to_rcn(gt_instances_s1)
            proposals_rpn_s2 = self.convert_gt_to_rcn(gt_instances_s2)

            gt_labels_s1 = gt_instances_s1[0].gt_classes
            gt_labels_s2 = gt_instances_s2[0].gt_classes

            for i , x in enumerate(gt_instances_s1):
                if (i != 0):
                    gt_labels_s1 = torch.cat((gt_labels_s1, x.gt_classes))
            for i, x in enumerate(gt_instances_s2):
                if (i != 0):
                    gt_labels_s2 = torch.cat((gt_labels_s2, x.gt_classes))


            box_features_s1 = self.roi_heads(
                images_s1,
                features_s1,
                proposals_rpn_s1,
                compute_loss=True,
                targets=gt_instances_s1,
                branch=branch,
            )

            box_features_s2 = self.roi_heads(
                images_s2,
                features_s2,
                proposals_rpn_s2,
                compute_loss=True,
                targets=gt_instances_s2,
                branch=branch,
            )
            if (self.prototype_layer):
                box_features_s1 = self.proto(box_features_s1)
                box_features_s2 = self.proto(box_features_s2)


            for lab, pro in zip(gt_labels_s1, box_features_s1):
                self.prototype_c[lab] = ((self.prototype_c[lab] * self.number_of_occurance_c[lab]) + pro)/(self.number_of_occurance_c[lab] + 1)
                self.number_of_occurance_c[lab] += 1

            for lab, pro in zip(gt_labels_s2, box_features_s2):
                self.prototype_c[lab] = ((self.prototype_c[lab] * self.number_of_occurance_c[lab]) + pro)/(self.number_of_occurance_c[lab] + 1)
                self.number_of_occurance_c[lab] += 1
            
            loss = 0
            for i, j in itertools.combinations(list(range(len(self.prototype_s1))),2):
                loss += F.cosine_similarity(self.prototype_s1[i].view(-1,1), self.prototype_s1[j].view(-1,1)).abs().mean().to(self.device)
            loss = loss/math.comb(len(self.prototype_s1), 2)
            self.prototype_s1.detach_()
            return loss

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
                
            source_label = target_type
            criterion = torch.nn.CrossEntropyLoss()
            loss_D_img_s = criterion(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))/1000.0
             
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            _, detector_losses, box_features = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )
            if (self.prototype_layer):
                box_features = self.proto(box_features)
                t_loss = F.cosine_similarity(box_features, box_features).abs().mean().to(self.device) *0
            else:
                t_loss = 0
            #print("below roi")
            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            
            losses["loss_D_img_s"] = loss_D_img_s * 0.0001 + t_loss
            return losses, [], [], None

        elif branch.startswith("prototype_s") or branch.startswith("prototype_t"):
            proposals_rpn = self.convert_gt_to_rcn(gt_instances)
            gt_labels = gt_instances[0].gt_classes
            for i, x in enumerate(gt_instances):
                if (i != 0):
                    gt_labels = torch.cat((gt_labels, x.gt_classes))
            
            _, detector_losses, box_features = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )
            if (self.prototype_layer):
                box_features = self.proto(box_features)

            if branch == "prototype_s":
                if (target_type == 1):
                    for lab, pro in zip(gt_labels, box_features):
                        self.prototype_s1[lab] = ((self.prototype_s1[lab] * self.number_of_occurance_s1[lab]) + pro)/(self.number_of_occurance_s1[lab] + 1)
                        self.number_of_occurance_s1[lab] += 1

                elif (target_type == 2):
                    for lab, pro in zip(gt_labels, box_features):
                        self.prototype_s2[lab] = ((self.prototype_s2[lab] * self.number_of_occurance_s2[lab]) + pro)/(self.number_of_occurance_s2[lab] + 1)
                        self.number_of_occurance_s2[lab] += 1
                
            if branch == "prototype_t":
                self.prototype_t.to(self.device)
                for lab, pro in zip(gt_labels, box_features):
                    self.prototype_t[lab] = ((self.prototype_t[lab] * self.number_of_occurance_t[lab]) + pro)/(self.number_of_occurance_t[lab] + 1)
                    self.number_of_occurance_t[lab] += 1
            
            loss_pro = F.cosine_similarity(self.prototype_s1, self.prototype_t).abs().mean().to(self.device)
            losses = {}
            losses["prototype"] = loss_pro
            return losses

        elif branch == "supervised_target":

            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses, _ = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()


    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch



@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses, _ = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses, _ = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None
    