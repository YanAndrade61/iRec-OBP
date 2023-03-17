from obp.dataset import SyntheticBanditDataset
from obp.types import BanditFeedback

class NewSyntheticBanditDataset(SyntheticBanditDataset):
  
  user_context_file: str
  
  def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
        """Obtain batch logged bandit data.

        Parameters
        ----------
        n_rounds: int
            Data size of the synthetic logged bandit data.

        Returns
        ---------
        bandit_feedback: BanditFeedback
            Synthesized logged bandit data.

        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        users = None
        if ml:
            users,contexts = select_context(n_rounds,self.n_actions)
        else:
            contexts = self.random_.normal(size=(n_rounds, self.dim_context))

        # calc expected reward given context and action
        expected_reward_ = self.calc_expected_reward(contexts)
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            # correct expected_reward_, as we use truncated normal distribution here
            mean = expected_reward_
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            expected_reward_ = truncnorm.stats(
                a=a, b=b, loc=mean, scale=self.reward_std, moments="m"
            )

        # calculate the action choice probabilities of the behavior policy
        if self.behavior_policy_function is None:
            pi_b_logits = expected_reward_
        else:
            pi_b_logits = self.behavior_policy_function(
                context=contexts,
                action_context=self.action_context,
                random_state=self.random_state,
            )
        # create some deficient actions based on the value of `n_deficient_actions`
        if self.n_deficient_actions > 0:
            pi_b = np.zeros_like(pi_b_logits)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                self.random_.gumbel(size=(n_rounds, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(n_rounds), (n_supported_actions, 1)).T,
                supported_actions,
            )
            pi_b[supported_actions_idx] = softmax(
                self.beta * pi_b_logits[supported_actions_idx]
            )
        else:
            pi_b = softmax(self.beta * pi_b_logits)
        # sample actions for each round based on the behavior policy
        if ml:
            actions = sample_action_fast_ml(pi_b,users, random_state=self.random_state)
        else:
            actions = sample_action_fast(pi_b, random_state=self.random_state)

        # sample rewards based on the context and action
        rewards = self.sample_reward_given_expected_reward(expected_reward_, actions)

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            context=contexts,
            action_context=self.action_context,
            users = users,
            action=actions,
            position=None,  # position effect is not considered in synthetic data
            reward=rewards,
            expected_reward=expected_reward_,
            pi_b=pi_b[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
        )
