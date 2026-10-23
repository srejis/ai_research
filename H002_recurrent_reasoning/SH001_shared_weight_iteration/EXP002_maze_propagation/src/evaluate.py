"""Final-target quality and algorithm-step fidelity are deliberately separate."""
import time
import numpy as np
import torch


def binary_bce(probability, target):
    p, y = probability.astype(np.float64), target.astype(np.float64)
    with np.errstate(divide='ignore'):
        log_p, log_not_p = np.maximum(np.log(p), -100), np.maximum(np.log1p(-p), -100)
    return float(-(y*log_p + (1-y)*log_not_p).mean())


def scores(pred, target):
    intersection = (pred & target).sum(1)
    union = (pred | target).sum(1)
    denominator = pred.sum(1) + target.sum(1)
    return {'macro_iou':float((intersection / np.maximum(union,1)).mean()),
            'macro_f1':float((2*intersection / np.maximum(denominator,1)).mean()),
            'pixel_error':float((pred != target).mean()),
            'exact_map_accuracy':float((pred == target).all(1).mean())}


def metrics_from_states(states, walls, goal, distances, start, mode):
    if not np.isfinite(states).all():
        raise FloatingPointError('Nonfinite evaluation states')
    assert states.shape[2] == 192
    final = distances >= 0
    disconnected = (walls == 0) & ~final
    previous_state = np.concatenate((walls,goal,goal),axis=1)
    previous_prediction = goal.astype(bool)
    effective_states = []
    rows = []
    for index,state in enumerate(states):
        k = index+1
        probability = state[:,128:]
        prediction = probability >= .5
        step_target = (distances >= 0) & (distances <= k)
        final_scores, step_scores = scores(prediction,final), scores(prediction,step_target)
        effective_state = state if mode in 'AB' else np.concatenate((walls,goal,probability),axis=1)
        change = np.abs(effective_state-previous_state)
        r_change = change[:,128:]
        new = prediction & ~previous_prediction
        removed = previous_prediction & ~prediction
        p = np.clip(probability.astype(np.float64),1e-12,1-1e-12)
        entropy = -(p*np.log(p)+(1-p)*np.log1p(-p))
        incorrect = prediction != final
        confident = (probability >= .9) | (probability <= .1)
        input_context = previous_state[:,:128] if mode in 'AB' else np.concatenate((walls,goal),axis=1)
        cycle_error = np.abs(effective_state-effective_states[index-2]).mean(1) if index>=2 else np.full(len(state),np.inf)
        start_prediction = prediction[np.arange(len(start)),start]
        start_actual = final[np.arange(len(start)),start]
        previous_error = (previous_prediction != final).mean(1)
        current_error = incorrect.mean(1)
        rows.append({'step':k, 'final_bce':binary_bce(probability,final), **final_scores,
                     'step_bce':binary_bce(probability,step_target),
                     **{'step_'+key:value for key,value in step_scores.items()},
                     'premature_reachable_cells_mean':float((prediction & final & ~step_target).sum(1).mean()),
                     'step_target_fnr':float((~prediction & step_target).sum()/max(step_target.sum(),1)),
                     'start_accuracy':float((start_prediction==start_actual).mean()),
                     'start_reachable_recall':float(start_prediction[start_actual].mean()),
                     'start_unreachable_specificity':float((~start_prediction[~start_actual]).mean()),
                     'unreachable_open_fpr':float((prediction & disconnected).sum()/max(disconnected.sum(),1)),
                     'reachable_fnr':float((~prediction & final).sum()/max(final.sum(),1)),
                     'wall_fpr':float((prediction & (walls==1)).sum()/max(walls.sum(),1)),
                     'state_change_mean':float(change.mean()), 'state_change_max':float(change.max()),
                     'r_change_mean':float(r_change.mean()), 'r_change_max':float(r_change.max()),
                     'predicted_area_mean':float(prediction.sum(1).mean()),
                     'new_area_mean':float(new.sum(1).mean()),
                     'correct_new_area_mean':float((new & final).sum(1).mean()),
                     'incorrect_new_area_mean':float((new & ~final).sum(1).mean()),
                     'removed_area_mean':float(removed.sum(1).mean()),
                     'correct_area_removed_mean':float((removed & final).sum(1).mean()),
                     'area_shrink_sample_fraction':float((prediction.sum(1)<previous_prediction.sum(1)).mean()),
                     'improved_sample_fraction':float((current_error<previous_error).mean()),
                     'worsened_sample_fraction':float((current_error>previous_error).mean()),
                     'entropy_nats':float(entropy.mean()),
                     'high_confidence_wrong_fraction_all_cells':float((confident & incorrect).mean()),
                     'high_confidence_fraction_among_wrong':float((confident & incorrect).sum()/max(incorrect.sum(),1)),
                     'fixed_point_fraction':float((change.mean(1)<1e-5).mean()),
                     'period2_fraction':float(((cycle_error<1e-5)&(change.mean(1)>1e-4)).mean()),
                     'raw_w_output_mae':float(np.abs(state[:,:64]-walls).mean()),
                     'raw_g_output_mae':float(np.abs(state[:,64:128]-goal).mean()),
                     'effective_input_w_mae':float(np.abs(input_context[:,:64]-walls).mean()),
                     'effective_input_g_mae':float(np.abs(input_context[:,64:128]-goal).mean()),
                     'raw_goal_argmax_accuracy':float((state[:,64:128].argmax(1)==goal.argmax(1)).mean()),
                     'nonfinite':0, 'outside_unit_interval':int(((state<0)|(state>1)).sum())})
        effective_states.append(effective_state)
        previous_state,previous_prediction = effective_state,prediction
    return rows


@torch.no_grad()
def evaluate(model,tensors,mode,steps):
    model.eval()
    states = torch.stack(model(tensors['walls'],tensors['goal'],steps,mode)).cpu().numpy()
    data = {k:v.cpu().numpy() for k,v in tensors.items()}
    metrics = metrics_from_states(states,data['walls'],data['goal'],data['distances'],data['start'],mode)
    return metrics,states


@torch.no_grad()
def latency(model,host_w,host_g,device,mode,config):
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    model.eval()
    w,g = host_w.to(device),host_g.to(device)
    rows = []
    for steps in range(1,config['eval_steps']+1):
        for _ in range(config['latency_warmup']):
            model(w,g,steps,mode)
        sync()
        row = {'step':steps,'batch_size':len(w)}
        for kind in ('model_only','end_to_end'):
            times = []
            for _ in range(config['latency_repeats']):
                sync()
                started = time.perf_counter()
                if kind == 'end_to_end':
                    out = model(host_w.to(device),host_g.to(device),steps,mode)[-1].cpu()
                else:
                    out = model(w,g,steps,mode)[-1]
                sync()
                times.append((time.perf_counter()-started)*1000)
            row[kind+'_ms_mean'] = float(np.mean(times))
            row[kind+'_ms_std'] = float(np.std(times))
        # 2 FLOPs per multiply-add; biases/activations/cat excluded.
        row['linear_forward_flops_per_sample'] = 2*(192*256+256*192)*steps
        rows.append(row)
    return rows
