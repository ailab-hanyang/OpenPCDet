import time
import numpy as np
import matplotlib.pyplot as plt

class CustomEval:
    def __init__(self, classes=['Regular_vehicle'],
                 distance_threshold=1.0,
                 min_score=0.0,
                 max_range=150.0):

        # Initialize
        self.distance_threshold_sq = distance_threshold**2
        self.score_threshold = min_score
        self.max_range = max_range
        self.classes = classes
        self.total_N_pos = 0
        self.results_dict = {}
        for single_class in classes:
            class_dict = {}
            class_dict['class'] = single_class
            class_dict['T_p'] = np.empty((0, 8))
            class_dict['gt'] = np.empty((0, 7))
            class_dict['total_N_pos'] = 0
            class_dict['result'] = np.empty((0, 2))
            class_dict['precision'] = []
            class_dict['recall'] = []
            self.results_dict[single_class] = class_dict
        
        # Start time
        self.time = time.time()

    def evaluate(self, dt_annos, gt_annos, class_names):
        
        ret_string = str()
        ret_dict = dict()
        # ## Check missing files
        # assert len(eval_det_annos) == len(eval_gt_annos)
        
        ## Debug
        if len(dt_annos) != len(gt_annos):
            gt_annos = gt_annos[:len(dt_annos)]

        print("Starting evaluation for {} file predictions".format(len(gt_annos)))
        print("--------------------------------------------")

        ## Evaluate matches
        print("Evaluation examples")
        
        for current_class in class_names:
            for i in range (len(gt_annos)):
                self.eval_pair(dt_annos[i], gt_annos[i], current_class)

            print("\nDone!")
            print("----------------------------------")

            ## Calculate
            for single_class in self.classes:
                class_dict = self.results_dict[single_class]
                print("Calculating metrics for {} class".format(single_class))
                print("----------------------------------")
                print("Number of ground truth labels: ", class_dict['total_N_pos'])
                print("Number of detections:  ", class_dict['result'].shape[0])
                print("Number of true positives:  ", np.sum(class_dict['result'][:, 0] == 1))
                print("Number of false positives:  ", np.sum(class_dict['result'][:, 0] == 0))
                if class_dict['total_N_pos'] == 0:
                    print("No detections for this class!")
                    print(" ")
                    continue
                ## AP
                self.compute_ap_curve(class_dict)
                mean_ap = self.compute_mean_ap(class_dict['precision'], class_dict['recall'])
                # print('Mean AP: %.3f ' % mean_ap)
                f1 = self.compute_f1_score(class_dict['precision'], class_dict['recall'])
                # print('F1 Score: %.3f ' % f1)
                # print(' ')
                # ATE 2D
                ate2d = self.compute_ate2d(class_dict['T_p'], class_dict['gt'])
                # print('Average 2D Translation Error [m]:  %.4f ' % ate2d)
                # ATE 3D
                ate3d = self.compute_ate3d(class_dict['T_p'], class_dict['gt'])
                # print('Average 3D Translation Error [m]:  %.4f ' % ate3d)
                # ASE
                ase = self.compute_ase(class_dict['T_p'], class_dict['gt'])
                # print('Average Scale Error:  %.4f ' % ase)
                # AOE
                aoe = self.compute_aoe(class_dict['T_p'], class_dict['gt'])
                # print('Average Orientation Error [rad]:  %.4f ' % aoe)
                # print(" ")

                # ret_dict = {
                #     'mean_ap': mean_ap,
                #     'f1_score': f1,
                #     'ate2d': ate2d,
                #     'ate3d': ate3d,
                #     'ase': ase,
                #     'aoe': aoe
                # }
                
                ret_dict['%s/AP' % single_class] = mean_ap
                ret_dict['%s/f1_score' % single_class] = f1
                # ret_dict['%s/precision' % single_class] = class_dict['precision']
                # ret_dict['%s/recall' % single_class] = class_dict['recall']
                ret_dict['%s/ate2d' % single_class] = ate2d
                ret_dict['%s/ate3d' % single_class] = ate3d
                ret_dict['%s/ase' % single_class] = ase
                ret_dict['%s/aoe' % single_class] = aoe

                ret_string += f"{current_class}\n"
                ret_string += f"AP:{mean_ap:.4f}\n"
                # results_string += f"Precision:{class_dict['precision']:.4f}"
                # results_string += f"Recall:{class_dict['recall']:.4f}"
                ret_string += f"F1 Score: {f1:.4f}\n"
                ret_string += f"ATE 2D: {ate2d:.4f}\n"
                ret_string += f"ATE 3D: {ate3d:.4f}\n"
                ret_string += f"ASE: {ase:.4f}\n"
                ret_string += f"AOE: {aoe:.4f}\n"

            self.time = float(time.time() - self.time)
            print("Total evaluation time: %.5f " % self.time)
        return ret_string, ret_dict

    def compute_ap_curve(self, class_dict):
        t_pos = 0
        class_dict['precision'] = np.ones(class_dict['result'].shape[0]+2)
        class_dict['recall'] = np.zeros(class_dict['result'].shape[0]+2)
        sorted_detections = class_dict['result'][(-class_dict['result'][:, 1]).argsort(), :]
        for i, (result_bool, result_score) in enumerate(sorted_detections):
            if result_bool == 1:
                t_pos += 1
            class_dict['precision'][i+1] = t_pos / (i + 1)
            class_dict['recall'][i+1] = t_pos / class_dict['total_N_pos']
        class_dict['precision'][i+2] = 0
        class_dict['recall'][i+2] = class_dict['recall'][i+1]

        # ## Plot
        # plt.figure()
        # plt.plot(class_dict['recall'], class_dict['precision'])
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision Recall curve for {} Class'.format(class_dict['class']))
        # plt.xlim([0, 1])
        # plt.ylim([0, 1.05])
        # plt.savefig(class_dict['class'] + "_pr_curve.png")

    def compute_f1_score(self, precision, recall):
        p, r = precision[(precision+recall) > 0], recall[(precision+recall) > 0]
        f1_scores = 2 * p * r / (p + r)
        return np.max(f1_scores)

    def compute_mean_ap(self, precision, recall, precision_threshold=0.0, recall_threshold=0.0):
        mean_ap = 0
        threshold_mask = np.logical_and(precision > precision_threshold,
                                        recall > recall_threshold)
        # calculate mean AP
        precision = precision[threshold_mask]
        recall = recall[threshold_mask]
        recall_diff = np.diff(recall)
        precision_diff = np.diff(precision)
        # Square area under curve based on i+1 precision, then linear difference in precision
        mean_ap = np.sum(precision[1:]*recall_diff + recall_diff*precision_diff/2)
        # We need to divide by (1-recall_threshold) to make the max possible mAP = 1. In practice threshold by the first
        # considered recall value (threshold = 0.1 -> first considered value may be = 0.1123)
        mean_ap = mean_ap/(1-recall[0])
        return mean_ap

    def compute_ate2d(self, predictions, ground_truth):
        # euclidean distance 3d
        mean_ate2d = np.mean(np.sqrt((predictions[:, 0] - ground_truth[:, 0])**2 +
                                     (predictions[:, 1] - ground_truth[:, 1])**2))
        return mean_ate2d

    def compute_ate3d(self, predictions, ground_truth):
        # euclidean distance 2d
        mean_ate3d = np.mean(np.sqrt((predictions[:, 0] - ground_truth[:, 0]) ** 2 +
                                     (predictions[:, 1] - ground_truth[:, 1]) ** 2 +
                                     (predictions[:, 2] - ground_truth[:, 2]) ** 2))
        return mean_ate3d

    def compute_ase(self, predictions, ground_truth):
        # simplified iou where boxes are centered and aligned with eachother
        pred_vol = predictions[:, 3]*predictions[:, 4]*predictions[:, 5]
        gt_vol = ground_truth[:, 3]*ground_truth[:, 4]*ground_truth[:, 5]
        iou3d = np.mean(1 - np.minimum(pred_vol, gt_vol)/np.maximum(pred_vol, gt_vol))
        return iou3d

    def compute_aoe(self, predictions, ground_truth):
        err = ground_truth[:,6] - predictions[:,6]
        aoe = np.mean(np.abs((err + np.pi) % (2*np.pi) - np.pi))
        return aoe

    def eval_pair(self, dt_anno, gt_anno, current_class):

        gt_label = np.hstack((gt_anno['name'].reshape(-1, 1), gt_anno['gt_boxes_lidar']))
        pred_label = np.hstack((dt_anno['name'].reshape(-1, 1), dt_anno['boxes_lidar'], dt_anno['score'].reshape(-1, 1)))

        # get all pred labels, order by score
        class_pred_label = pred_label[np.char.lower(pred_label[:, 0].astype(str)) == current_class.lower(), 1:]
        score = class_pred_label[:, 7].astype(np.float)
        class_pred_label = class_pred_label[(-score).argsort(), :].astype(np.float) # sort decreasing

        # add gt label length to total_N_pos
        class_gt_label = gt_label[np.char.lower(gt_label[:, 0].astype(str)) == current_class.lower(), 1:].astype(np.float)
        self.results_dict[current_class]['total_N_pos'] += class_gt_label.shape[0]

        # match pairs
        pred_array, gt_array, result_score_pair = self.match_pairs(class_pred_label, class_gt_label)

        # add to existing results
        self.results_dict[current_class]['T_p'] = np.vstack((self.results_dict[current_class]['T_p'], pred_array))
        self.results_dict[current_class]['gt'] = np.vstack((self.results_dict[current_class]['gt'], gt_array))
        self.results_dict[current_class]['result'] = np.vstack((self.results_dict[current_class]['result'],
                                                                result_score_pair))

    def match_pairs(self, pred_label, gt_label):
        true_preds = np.empty((0, 8))
        corresponding_gt = np.empty((0, 7))
        result_score = np.empty((0, 2))
        # Initialize matching loop
        match_incomplete = True
        while match_incomplete and gt_label.shape[0] > 0:
            match_incomplete = False
            for gt_idx, single_gt_label in enumerate(gt_label):
                # Check is any prediction is in range
                distance_sq_array = (single_gt_label[0] - pred_label[:, 0])**2 + (single_gt_label[1] - pred_label[:, 1])**2
                # If there is a prediction in range, pick closest
                if np.any(distance_sq_array < self.distance_threshold_sq):
                    min_idx = np.argmin(distance_sq_array)
                    # Store true prediction
                    true_preds = np.vstack((true_preds, pred_label[min_idx, :].reshape(-1, 1).T))
                    corresponding_gt = np.vstack((corresponding_gt, gt_label[gt_idx]))

                    # Store score for mAP
                    result_score = np.vstack((result_score, np.array([[1, pred_label[min_idx, 7]]])))

                    # Remove prediction and gt then reset loop
                    pred_label = np.delete(pred_label, obj=min_idx, axis=0)
                    gt_label = np.delete(gt_label, obj=gt_idx, axis=0)
                    match_incomplete = True
                    break

        # If there were any false detections, add them.
        if pred_label.shape[0] > 0:
            false_positives = np.zeros((pred_label.shape[0], 2))
            false_positives[:, 1] = pred_label[:, 7]
            result_score = np.vstack((result_score, false_positives))
        return true_preds, corresponding_gt, result_score

    def filter_by_range(self, pred_label, gt_label, range=0):
        pred_dist = np.linalg.norm(pred_label[:, 1:4].astype(np.float32), axis=1) < range
        gt_dist = np.linalg.norm(gt_label[:, 1:4].astype(np.float32), axis=1) < range
        return pred_label[pred_dist, :], gt_label[gt_dist, :]

