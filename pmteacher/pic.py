from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

evaluator = trainer.build_evaluator(cfg, "DuskDawn_val")
test_loader = trainer.build_test_loader(cfg, "DuskDawn_val")


def plot_detection_boxes(predictions, cluster_boxes, data_dict):
    img = Image.open(data_dict["file_name"])
    plt.figure(figsize=(20,10))
    plt.axis('off')
    plt.imshow(img)
    ax = plt.gca()    
    if len(predictions)!=0:
        predictions = predictions[predictions.scores>0.6]
        predictions = predictions.pred_boxes.tensor.cpu()
        for bbox in predictions:
            x1, y1 = bbox[0], bbox[1]
            h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='orange', facecolor='none')
            ax.add_patch(rect)
    '''if len(cluster_boxes)!=0:
        if isinstance(cluster_boxes, Instances):
            cluster_boxes = cluster_boxes.pred_boxes.tensor.cpu()
        for bbox in cluster_boxes:
            x1, y1 = bbox[0], bbox[1]
            h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    '''
    im_name = os.path.basename(data_dict["file_name"])[:-4]
    plt.savefig(os.path.join(os.getcwd(), 'temp', 'dota_comp', im_name+"_det_ss.jpg"), dpi=90, bbox_inches='tight')
    plt.savefig(os.path.join(os.getcwd(), 'dota_comp', im_name+"_det_base.jpg"), dpi=90, bbox_inches='tight')
    plt.clf()