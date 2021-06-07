
from nle import nethack
import torch
from torch import nn

from .common import *
from .distributions import Categorical  


class MiniHackNet(nn.Module):
    def __init__(
        self,
        input_shape,
        num_actions,
        obs_use,
        actor_fc_layers=(32, 32),
        value_fc_layers=(32, 32),
        glyph_embedding_dim=32,
        recurrent_arch='lstm',
        recurrent_hidden_size=256,
        num_layers=5,
    ):
        super(MiniHackNet, self).__init__()
        self.input_shape = input_shape

        self.num_actions = num_actions
        self.recurrent_arch = recurrent_arch

        self.H = self.input_shape[0]
        self.W = self.input_shape[1]

        self.k_dim = glyph_embedding_dim
        self.h_dim = recurrent_hidden_size

        self.obs_use = obs_use

        self.embed = nn.Embedding(nethack.MAX_GLYPH, glyph_embedding_dim)

        self.image_conv = nn.Sequential(
            Conv2d_tf(glyph_embedding_dim, 16, kernel_size=3, stride=1, padding='VALID'),
            nn.ReLU(),
            Conv2d_tf(16, 32, kernel_size=3, stride=2, padding='VALID'),
            nn.Flatten(),
            nn.ReLU()
        )
        if obs_use == 'image':
            self.image_embedding_size = 800
        elif obs_use == 'chars_crop':
            self.image_embedding_size = 32

        self.rnn = None
        if recurrent_arch:
            self.rnn = RNN(
                input_size=self.image_embedding_size,
                hidden_size=self.h_dim,
                arch=recurrent_arch)
            self.base_output_size = self.h_dim
            self.recurrent_hidden_state_size = self.h_dim
        else:
            self.base_output_size = self.image_embedding_size
            self.recurrent_hidden_state_size = 0

        self.actor = nn.Sequential(
            make_fc_layers_with_hidden_sizes(actor_fc_layers, input_size=self.base_output_size),
            Categorical(actor_fc_layers[-1], num_actions)
        )
        self.critic = nn.Sequential(
            make_fc_layers_with_hidden_sizes(value_fc_layers, input_size=self.base_output_size),
            init_(nn.Linear(value_fc_layers[-1], 1))
        )

    @property
    def is_recurrent(self):
        return self.rnn is not None

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def _forward_base(self, inputs, rnn_hxs, masks):
        image = inputs.get(self.obs_use)

        image_emb = self._select(self.embed, image.long())
        image_emb = image_emb.transpose(1, 3)  # -- TODO: slow?
        image_emb = self.image_conv(image_emb)

        # in_features = [image_emb]
        # in_features = torch.cat(in_features, dim=1)
        in_features = image_emb

        if self.recurrent_arch:
            core_features, rnn_hxs = self.rnn(in_features, rnn_hxs, masks)
        else:
            core_features = in_features

        return core_features, rnn_hxs

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        dist = self.actor(core_features)
        value = self.critic(core_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_dist = dist.logits
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)
        return self.critic(core_features)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        core_features, rnn_hxs = self._forward_base(inputs, rnn_hxs, masks)

        dist = self.actor(core_features)
        value = self.critic(core_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs