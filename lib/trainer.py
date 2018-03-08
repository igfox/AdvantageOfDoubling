import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import argparse
import copy
import time


def to_cuda(x, device=1):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            if type(x) is list:
                return [item.cuda() for item in x]
            return x.cuda()
    return x


def to_variable(x):
    if type(x) is list:
        return [Variable(item).float() for item in x]
    return Variable(x).float()


def to_np(x):
    return x.data.cpu().numpy()


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", help="root directory for this run",
                        default=None)
    parser.add_argument("-g", "--device", help="gpu device number, default to 1", type=int,
                        default=1)
    parser.add_argument('--name', help='name for saving run', default=time.time())
    parser.add_argument('--type', help='network type: TD, DQN, DDQN', choices=["onp", "dqn", "ddqn"])
    return parser


class DeepQTrainer:
    """
    A Deep Q Network trainer. Supports TD learning, Q learning, and Double Q learning
    """

    def __init__(self,
                 data,
                 batch_size,
                 epoch_limit,
                 criterion,
                 save_loc,
                 name,
                 log_dir,
                 gpu_device,
                 lr,
                 clip,
                 on_policy,
                 double_q,
                 num_workers,
                 reset_rate,
                 validate_rate):
        """
        :param data: A pytorch Dataset
        :param batch_size: training/validation batch size
        :param epoch_limit: number of training epochs
        :param criterion: training loss
        :param save_loc: save directory
        :param name: experiment name
        :param log_dir: save directory for tensorboard log files
        :param gpu_device: numbered gpu device on which to run
        :param lr: learning rate for ADAM optimizer
        :param clip: parameter for gradient clipping
        :param on_policy: boolean switch to use TD learning instead of Q learning
        :param double_q: boolean switch to use double Q learning instead of Q learning
        :param num_workers: number of data-loading threads to use
        :param reset_rate: rate to cache target network
        :param validate_rate: rate to check validation performance
        """
        self.writer = SummaryWriter(log_dir='{}/{}'.format(log_dir, name))
        # set various attributes
        attribute_dict = {'epoch_limit': epoch_limit,
                          'save_loc': save_loc, 'batch_size': batch_size, 'criterion': criterion, 'name': name,
                          'gpu_device': gpu_device, 'lr': lr, 'clip': clip, 'data': data,
                          'on_policy': on_policy, 'double_q': double_q, 'reset_rate': reset_rate,
                          'validate_rate': validate_rate}
        for key in attribute_dict:
            setattr(self, key, attribute_dict[key])
        self.total_steps = 0
        self.model = None
        self.old_model = None

        sampler = torch.utils.data.sampler.SubsetRandomSampler
        train_sampler = sampler(data.train_ind)
        valid_sampler = sampler(data.valid_ind)
        test_sampler = sampler(data.test_ind)

        self.train_data = DataLoader(data,
                                     batch_size=batch_size,
                                     sampler=train_sampler,
                                     num_workers=num_workers)
        self.train_data = DataLoader(data,
                                     batch_size=batch_size,
                                     sampler=train_sampler,
                                     num_workers=num_workers)
        self.valid_data = DataLoader(data,
                                     batch_size=batch_size,
                                     sampler=valid_sampler,
                                     num_workers=num_workers)
        self.test_data = DataLoader(data,
                                    batch_size=batch_size,
                                    sampler=test_sampler,
                                    num_workers=num_workers)

    def time_to_reset(self):
        return self.total_steps % self.reset_rate == 0

    def time_to_validate(self):
        return self.total_steps % self.validate_rate == 0

    def train(self, model, gamma, optimizer=None):
        """
        Training loop
        :param model: pytorch model to be trained
        :param gamma: discount factor
        :param optimizer: training optimizer, defaults to ADAM
        :return:
        """
        print('starting train')
        torch.save(model.state_dict(), '{}/{}_start.pt'.format(self.save_loc, self.name))
        self.model = to_cuda(model, self.gpu_device)
        self.old_model = copy.deepcopy(model)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=self.lr)

        for epoch_num in range(self.epoch_limit):
            print('epoch {}'.format(epoch_num))
            running_train_loss = 0
            for state, action, reward, next_state, next_action, feasible_mask in self.train_data:
                if self.time_to_validate():
                    print('validating')
                    self.validate(gamma=gamma)
                if self.time_to_reset():
                    print('saving model at epoch {}, step {}'.format(epoch_num, self.total_steps))
                    self.old_model = copy.deepcopy(self.model)
                    torch.save(self.model.state_dict(),
                               '{}/{}_epoch_{}_step_{}.pt'.format(self.save_loc, self.name, epoch_num,
                                                                  self.total_steps))

                self.total_steps += 1
                train_loss = self.optimize(state=state,
                                           action=action,
                                           reward=reward,
                                           next_state=next_state,
                                           next_action=next_action,
                                           gamma=gamma,
                                           optimizer=optimizer,
                                           feasible_mask=feasible_mask)
                running_train_loss += train_loss
            self.writer.add_scalar(tag='data/train_epoch_loss',
                                   scalar_value=running_train_loss / len(self.train_data),
                                   global_step=self.total_steps)
        torch.save(self.model.state_dict(), '{}/{}_final.pt'.format(self.save_loc, self.name))

    def validate(self, gamma):
        """
        Evaluates on validation set. Used to monitor training performance
        :param gamma: discount factor
        :return:
        """
        self.model.eval()
        running_valid_loss = 0
        for state, action, reward, next_state, next_action, feasible_mask in self.valid_data:
            valid_loss = self.optimize(state=state,
                                       action=action,
                                       reward=reward,
                                       next_state=next_state,
                                       next_action=next_action,
                                       gamma=gamma,
                                       optimizer=None,
                                       feasible_mask=feasible_mask,
                                       valid=True)
            running_valid_loss += valid_loss
        self.writer.add_scalar(tag='data/valid_epoch_loss',
                               scalar_value=running_valid_loss / len(self.valid_data),
                               global_step=self.total_steps)
        print('Validation loss: {:.2f}'.format(running_valid_loss / len(self.valid_data)))
        self.model.train()

    def _q_state_value_estimate(self, state, nonterminal_mask, non_final_states, feasible_mask):
        """
        Helper function which calculates the target state value using a Q value estimation procedure
        :param state: a state tensor
        :param nonterminal_mask: a boolean mask denoting the terminal states
        :param non_final_states: the next state for all non-terminal states
        :param feasible_mask: a boolean mask indicating which actions are allowed in the current state
        :return: An estimate of state value, using the maximal Q values derived from self.old_model
        """
        next_state_values = to_variable(to_cuda(torch.zeros(state[0].size(0)).float(), self.gpu_device))
        nonterminal_feasible_mask = feasible_mask[nonterminal_mask.nonzero().view(-1)]
        predictions = self.old_model(to_cuda(non_final_states, self.gpu_device))
        # modifying predictions by adjusted to ensure max value is within feasible action set
        adjuster = 2 * max(abs(predictions.min().data[0]), predictions.max().data[0])
        adjusted_predictions = predictions - adjuster
        adjusted_predictions[nonterminal_feasible_mask] += adjuster
        next_state_values[nonterminal_mask] = adjusted_predictions.max(1)[0]
        next_state_values.volatile = False
        return next_state_values

    def _double_q_state_value_estimate(self, state, nonterminal_mask, non_final_states, feasible_mask):
        """
        Helper function which calculates the target state value using a Double Q value estimation procedure.
        Essentially the same as _q_state_value_estimate, except we use the current network to choose the actions
        which inform next state value.
        :param state: a state tensor
        :param nonterminal_mask: a boolean mask denoting the terminal states
        :param non_final_states: the next state for all non-terminal states
        :param feasible_mask: a boolean mask indicating which actions are allowed in the current state
        :return: An estimate of state value, using the maximal Q values derived from self.old_model
        """
        next_state_values = to_variable(to_cuda(torch.zeros(state[0].size(0)).float(), self.gpu_device))
        nonterminal_feasible_mask = feasible_mask[nonterminal_mask.nonzero().view(-1)]
        predictions_dq = self.model(to_cuda(non_final_states, self.gpu_device))
        # modifying predictions by adjusted to ensure max value is within feasible action set
        adjuster = 2 * max(abs(predictions_dq.min().data[0]), predictions_dq.max().data[0])
        adjusted_predictions_dq = predictions_dq - adjuster
        adjusted_predictions_dq[nonterminal_feasible_mask] += adjuster
        max_vals_dq, max_inds_dq = adjusted_predictions_dq.max(1)
        predictions = self.old_model(to_cuda(non_final_states, self.gpu_device))
        next_state_values[nonterminal_mask] = predictions.gather(1, max_inds_dq.view(-1, 1))
        next_state_values.volatile = False
        return next_state_values

    def _on_policy_state_value_estimate(self, state, next_action, nonterminal_mask, non_final_states):
        """
        Helper function which calculates the target state value using a TD value estimation procedure.
        Essentially the same as _q_state_value_estimate, except we use the observed next action to choose the actions
        which inform next state value. Note this corresponds to on-policy TD value estimation.
        :param state: a state tensor
        :param next_action: the next action taken at non-terminal states
        :param nonterminal_mask: a boolean mask denoting the terminal states
        :param non_final_states: the next state for all non-terminal states
        :return: An estimate of state value, using the maximal Q values derived from self.old_model
        """
        next_state_values = Variable(to_cuda(torch.zeros(state[0].size(0)).float(), self.gpu_device))
        predictions = self.old_model(to_cuda(non_final_states, self.gpu_device))
        next_state_values[nonterminal_mask] = predictions.gather(1, Variable(
            to_cuda(next_action, self.gpu_device)[nonterminal_mask].view(-1, 1)))
        next_state_values.volatile = False
        return next_state_values

    def log_q_diff(self, pred):
        """
        A convenience function for logging Q value distribution. Can be helpful in diagnosing value collapse.
        :param pred: Predicted Q values
        :return:
        """
        pred_sample = pred.data.cpu().numpy()
        q_diffs = np.diff(np.percentile(pred_sample, [0, 25, 50, 75, 100], axis=1), axis=0)
        dim_names = ['0_25', '25_50', '50_75', '75_100']
        for dim in range(len(q_diffs)):
            self.writer.add_histogram(tag='predictions/q_diff_{}'.format(dim_names[dim]),
                                      values=q_diffs[dim],
                                      global_step=self.total_steps,
                                      bins='auto')

    def log_values_and_advantages(self, state):
        """
        Convenience function for logging state value and action-advantage values
        :param state: vector of states
        :return:
        """
        advantage = self.model.get_advantage(to_variable(to_cuda(state, self.gpu_device)))
        self.writer.add_histogram(tag='predictions/advantages',
                                  values=advantage.view(-1),
                                  global_step=self.total_steps,
                                  bins='auto')
        value = self.model.get_value(to_variable(to_cuda(state, self.gpu_device)))
        self.writer.add_histogram(tag='predictions/values',
                                  values=value.view(-1),
                                  global_step=self.total_steps,
                                  bins='auto')

    def optimize(self,
                 state, action, reward, next_state, next_action,
                 gamma, optimizer, feasible_mask, valid=False):
        """
        Runs a step of optimization in the training/validation loops
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param next_action:
        :param gamma:
        :param optimizer:
        :param feasible_mask:
        :param valid:
        :return:
        """
        # each state is a tuple of spatial and flat information
        next_state_court = next_state[0]
        next_state_flat = next_state[1]
        # get non final states
        nonterminal_mask = []
        for batch_id in range(next_state_court.shape[0]):
            nonterminal_mask.append(np.min(next_state_court[batch_id].numpy()))
        nonterminal_mask = to_cuda(torch.from_numpy(np.array(nonterminal_mask)) > 0, self.gpu_device)
        non_final_states_court = Variable(
            torch.cat(
                [next_state_court[batch_id].view(1,
                                                 next_state_court[batch_id].size(0),
                                                 next_state_court[batch_id].size(1),
                                                 next_state_court[batch_id].size(2))
                 for batch_id in range(len(next_state_court)) if nonterminal_mask[batch_id]],
                dim=0),
            volatile=True)
        non_final_states_flat = Variable(
            torch.cat(
                [next_state_flat[batch_id].view(1,
                                                next_state_flat[batch_id].size(0))
                 for batch_id in range(len(next_state_flat)) if nonterminal_mask[batch_id]],
                dim=0),
            volatile=True)
        non_final_states = [non_final_states_court, non_final_states_flat]
        # get q values for observed actions
        predictions = self.model(to_variable(to_cuda(state, self.gpu_device)))
        state_action_values = predictions.gather(1, Variable(to_cuda(action, self.gpu_device)))
        # for non-final states, get V(s'), 0 for terminal states
        feasible_mask = to_cuda(feasible_mask, self.gpu_device)
        if self.on_policy:
            next_state_values = self._on_policy_state_value_estimate(state=state,
                                                                     next_action=next_action,
                                                                     nonterminal_mask=nonterminal_mask,
                                                                     non_final_states=non_final_states)
        elif self.double_q:
            next_state_values = self._double_q_state_value_estimate(state=state,
                                                                    nonterminal_mask=nonterminal_mask,
                                                                    non_final_states=non_final_states,
                                                                    feasible_mask=feasible_mask)
        else:
            next_state_values = self._q_state_value_estimate(state=state,
                                                             nonterminal_mask=nonterminal_mask,
                                                             non_final_states=non_final_states,
                                                             feasible_mask=feasible_mask)
        # combine discounted next state values with reward to get expected state value
        expected_state_action_values = (next_state_values * gamma) + Variable(to_cuda(reward.view(-1), self.gpu_device))
        # loss is between expected and predicted state-action values
        loss = self.criterion(state_action_values, expected_state_action_values)
        if not valid:
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            total_grad = 0
            for param in self.model.parameters():
                param.grad.data.clamp_(-1 * self.clip, self.clip)
                total_grad += np.sum(np.abs(to_np(param.grad)))
            if self.total_steps % 10 == 0:
                self.writer.add_scalar(tag='data/train_loss',
                                       scalar_value=to_np(loss),
                                       global_step=self.total_steps)
                self.writer.add_scalar(tag='data/gradient',
                                       scalar_value=total_grad,
                                       global_step=self.total_steps)
            if self.total_steps % 1000 == 0:
                self.log_q_diff(predictions)
                self.log_values_and_advantages(state)
                self.writer.add_histogram(tag='predictions/q_taken',
                                          values=state_action_values,
                                          global_step=self.total_steps,
                                          bins='auto')
                self.writer.add_histogram(tag='predictions/qs',
                                          values=predictions.view(-1),
                                          global_step=self.total_steps,
                                          bins='auto')
            optimizer.step()
        return loss.cpu().data[0]
