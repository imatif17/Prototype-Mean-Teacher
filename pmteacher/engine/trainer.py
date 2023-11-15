import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators

from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from pmteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
    build_detection_semisup_train_loader_two_crops_multi,
    build_detection_semisup_train_loader_two_crops_multi2
)
from pmteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from pmteacher.engine.hooks import LossEvalHook
from pmteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from pmteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from pmteacher.solver.build import build_lr_scheduler
from pmteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator, BDDDetectionEvaluator

from .probe import OpenMatchTrainerProbe
import copy

class PMTTrainer(DefaultTrainer):
    def __init__(self, cfg):
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        source1 = self.build_train_loader(cfg, data_val = 0, super_only = True)
        source2 = self.build_train_loader(cfg, data_val = 1, super_only = True)
        #source3 = self.build_train_loader(cfg, data_val = 2, super_only = True)
        target = self.build_train_loader(cfg, data_val = 2, super_only = False)
        self.source1 = source1
        self.source2 = source2
        #self.source3 = source3
        self.target = target
        model = self.build_model(cfg)
        optimizer = build_optimizer(cfg, model)
        #self.optimizer = optimizer
        #self.data_l = data_loader
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, source2, optimizer
        )
        #self.model = model
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)
        #self._trainer = self.model
        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        if comm.get_world_size() > 1:
            self.model.module.build_prototype()
        else:
            self.model.build_prototype()
        self.model_teacher.build_prototype()
        self._data_loader_iter_s1 = None
        self._data_loader_iter_s2 = None
        #self._data_loader_iter_s3 = None
        self._data_loader_iter_t = None
        self.datas = [self._data_loader_iter_s1, self._data_loader_iter_s2]
        self.probe = OpenMatchTrainerProbe(cfg)
        self.register_hooks(self.build_hooks()) 
        
    @property
    def _data_loader_iter1(self):
        if self._data_loader_iter_s1 is None:
            self._data_loader_iter_s1 = iter(self.source1)
        return self._data_loader_iter_s1
    
    @property
    def _data_loader_iter2(self):
        if self._data_loader_iter_s2 is None:
            self._data_loader_iter_s2 = iter(self.source2)
        return self._data_loader_iter_s2
    
    @property
    def _data_loader_iter3(self):
        if self._data_loader_iter_s3 is None:
            self._data_loader_iter_s3 = iter(self.source3)
        return self._data_loader_iter_s3

    @property
    def _data_loader_iter4(self):
        if self._data_loader_iter_t is None:
            self._data_loader_iter_t = iter(self.target)
        return self._data_loader_iter_t

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
        if isinstance(self.model, DistributedDataParallel):
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
        self.start_iter = comm.all_gather(self.start_iter)[0]


    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        elif evaluator_type == "pascal_voc_bdd":
            return BDDDetectionEvaluator(dataset_name, target_classnames=['traffic light', 'traffic sign', 'car', 'person', 'bus', 'truck', 'rider', 'bike', 'motor', 'train'])
        elif evaluator_type == "pascal_voc_bdd2":
            return BDDDetectionEvaluator(dataset_name, target_classnames= cfg.DATALOADER.CLASS)
        elif evaluator_type == "pascal_voc_bdd_coco":
            return BDDDetectionEvaluator(dataset_name, target_classnames=['person', 'rider', 'car', 'truck', 'bus', 'motor', 'bike'])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)
    
    @classmethod
    def build_train_loader(cls, cfg, data_val = 0, super_only = True):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops_multi2(cfg, mapper, data_val, super_only)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter
        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()
            
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list               

    def teacher_predictions(self, unlabel_data_k, unlabel_data_q):
        with torch.no_grad():
            _,proposals_rpn_unsup_k, proposals_roih_unsup_k, _ = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

        cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

        joint_proposal_dict = {}
        joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
        (
        pesudo_proposals_rpn_unsup_k,
        nun_pseudo_bbox_rpn,
        ) = self.process_pseudo_label(
            proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
            )

        joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
        pesudo_proposals_roih_unsup_k, _ = self.process_pseudo_label(
            proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
        )
        joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k

        unlabel_data_q = self.add_label(
            unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
        )
        unlabel_data_k = self.add_label(
            unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
        )
        return unlabel_data_k, unlabel_data_q

    def run_step_full_semisup(self):
        data_t = next(self._data_loader_iter4)
        unlabel_data_q, unlabel_data_k = data_t
        t_losses = 0
        grad = 1
        all_data = []
        contra = []
        for i, _ in enumerate(self.datas):
            assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
            start = time.perf_counter()
            if (i == 0):
                data_s = next(self._data_loader_iter1)
            elif (i == 1):
                data_s = next(self._data_loader_iter2)
            elif (i == 2):
                data_s = next(self._data_loader_iter3)
            # data_q and data_k from different augmentations (q:strong, k:weak)
            # label_strong, label_weak, unlabed_strong, unlabled_weak
            label_data_q, label_data_k = data_s
            data_time = time.perf_counter() - start

            # burn-in stage (supervised training with labeled data)
            if (self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP):
                # input both strong and weak supervised data into model
                label_data_q.extend(label_data_k)
                record_dict, _, _, _ = self.model(
                    label_data_q, branch="supervised", target_type = i+1)
                if(self.cfg.SEMISUPNET.USE_CONTRA):
                    contra.append(label_data_q)
                # weight losses
                loss_dict = {}
                for key in record_dict.keys():
                    if key[:4] == "loss":
                        loss_dict[key] = record_dict[key] * 1
                losses = sum(loss_dict.values())

            elif (self.iter <= self.cfg.SEMISUPNET.PROTOTYPE_WARMUP and self.cfg.SEMISUPNET.USE_PROTOTYPE):
                grad = 0
                if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                    self._update_teacher_model(keep_rate=0.00)
                label_data_q.extend(label_data_k)
                with torch.no_grad():
                    loss = self.model(label_data_q, branch ="prototype_s", target_type = i+1)
                    losses = sum(loss.values())
                unlabel_data_q = self.remove_label(unlabel_data_q)
                unlabel_data_k = self.remove_label(unlabel_data_k)
                
                unlabel_data_k, unlabel_data_q = self.teacher_predictions(unlabel_data_k, unlabel_data_q)
                unlabel_data_q.extend(unlabel_data_k)
                with torch.no_grad():
                    _ = self.model(unlabel_data_q, branch ="prototype_t", target_type = i+1)
                record_dict = {}
                continue
            
            else:
                if (self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP) and (i == 0):
                    # update copy the the whole model
                    self._update_teacher_model(keep_rate=0.00)
                    # self.model.build_discriminator()
                elif ((
                    self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
                ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0) and (i == 0):
                    self._update_teacher_model(
                        keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

                record_dict = {}

                gt_unlabel_k = self.get_label(unlabel_data_k)                

                #  0. remove unlabeled data labels
                unlabel_data_q = self.remove_label(unlabel_data_q)
                unlabel_data_k = self.remove_label(unlabel_data_k)

                #  1. generate the pseudo-label using teacher model
                unlabel_data_k, unlabel_data_q = self.teacher_predictions(unlabel_data_k, unlabel_data_q)

                all_label_data = label_data_q + label_data_k
                all_unlabel_data = unlabel_data_q
                if (i==0):
                    all_data.append(unlabel_data_k)
                all_data.append(label_data_k)
                # 4. input both strongly and weakly augmented labeled data into student model
                record_all_label_data, _, _, _ = self.model(
                    all_label_data, branch="supervised", target_type = i+1
                )
                record_dict.update(record_all_label_data)

                # 5. input strongly augmented unlabeled data into model
                record_all_unlabel_data, _, _, _ = self.model(
                    all_unlabel_data, branch="supervised_target",  target_type = i+1
                )
                new_record_all_unlabel_data = {}
                for key in record_all_unlabel_data.keys():
                    new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                        key
                    ]
                record_dict.update(new_record_all_unlabel_data)

                # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
                # give sign to the target data

                for i_index in range(len(unlabel_data_k)):
                    # unlabel_data_item = {}
                    for k, v in unlabel_data_k[i_index].items():
                        # label_data_k[i_index][k + "_unlabeled"] = v
                        label_data_k[i_index][k + "_unlabeled"] = v
                    # unlabel_data_k[i_index] = unlabel_data_item

                all_domain_data = label_data_k
                # all_domain_data = label_data_k + unlabel_data_k
                record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain", target_type = i+1)
                record_dict.update(record_all_domain_data)


                # weight losses
                loss_dict = {}
                for key in record_dict.keys():
                    if key.startswith("loss"):
                        if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                            # pseudo bbox regression <- 0
                            loss_dict[key] = record_dict[key] * 0
                        elif key[-6:] == "pseudo":  # unsupervised loss
                            loss_dict[key] = (
                                record_dict[key] *
                                self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                            )
                        elif (
                            key == "loss_D_img_s" or key == "loss_D_img_t"
                        ):  # set weight for discriminator
                            # import pdb
                            # pdb.set_trace()
                            loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
                        else:  # supervised loss
                            loss_dict[key] = record_dict[key] * 1

                losses = sum(loss_dict.values())
            t_losses += losses
        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        if (grad and self.iter>self.cfg.SEMISUPNET.PROTOTYPE_WARMUP):
            if (self.iter >= self.cfg.SEMISUPNET.PROTOTYPE_WARMUP and self.cfg.SEMISUPNET.USE_PROTOTYPE):
                pro_loss, contra_loss = self.model(all_data, branch="prototype_all", target_type = 1)
                metrics_dict["loss_prototype"] = pro_loss * self.cfg.SEMISUPNET.PROTO_WEIGHT
                metrics_dict["loss_contra"] = contra_loss * self.cfg.SEMISUPNET.CONTRA_WEIGHT
                t_losses += pro_loss * self.cfg.SEMISUPNET.PROTO_WEIGHT
                t_losses += contra_loss * self.cfg.SEMISUPNET.CONTRA_WEIGHT

        self._write_metrics(metrics_dict)
        self.optimizer.zero_grad()
        if grad:
            t_losses.backward()
        self.optimizer.step()


    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret