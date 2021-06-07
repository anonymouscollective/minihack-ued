from algos import PPO, RolloutStorage, ACAgent
from models import MiniHackNet


def model_for_adversarial_env(
    args,
    env,
    agent_type='agent',
    recurrent_arch=None):

    if agent_type == 'adversary_env':
        adversary_observation_space = env.adversary_observation_space
        adversary_obs_shape = adversary_observation_space['image'].shape
        adversary_num_actions = env.adversary_action_space.n
        adversary_max_timestep = adversary_observation_space['time_step'].high[0] + 1
        adversary_random_z_dim = adversary_observation_space['random_z'].shape[0]
    else:
        observation_space = env.observation_space
        num_actions = env.action_space.n

    if agent_type == 'adversary_env':
        model = MiniHackNet(input_shape=adversary_obs_shape,
                    num_actions=adversary_num_actions,
                    recurrent_arch=None,
                    obs_use='image')

    elif agent_type == 'agent':
        model = MiniHackNet(input_shape=observation_space['chars_crop'].shape,
               num_actions=num_actions,
               recurrent_arch='lstm',
               obs_use='chars_crop')

    elif agent_type == 'adversary_agent':
        model = MiniHackNet(input_shape=observation_space['chars_crop'].shape,
               num_actions=num_actions,
               recurrent_arch='lstm',
               obs_use='chars_crop')
    else:
        raise ValueError(f'Unsupported agent type {agent_type}')

    return model


def make_agent(name, env, args, device='cpu'):
    # Create model instance
    is_adversary_env = 'env' in name

    if is_adversary_env:
        observation_space = env.adversary_observation_space
        action_space = env.adversary_action_space
        num_steps = observation_space['time_step'].high[0]
        recurrent_arch = args.recurrent_adversary_env and args.recurrent_arch
        entropy_coef = args.adv_entropy_coef
    else:
        observation_space = env.observation_space
        action_space = env.action_space
        num_steps = args.num_steps
        recurrent_arch = args.recurrent_agent and args.recurrent_arch
        entropy_coef = args.entropy_coef

    actor_critic = model_for_adversarial_env(args, env, name, recurrent_arch=recurrent_arch)

    algo = None
    storage = None
    agent = None

    if args.algo == 'ppo':
        # Create PPO
        algo = PPO(
            actor_critic=actor_critic,
            clip_param=args.clip_param,
            ppo_epoch=args.ppo_epoch,
            num_mini_batch=args.num_mini_batch,
            value_loss_coef=args.value_loss_coef,
            entropy_coef=entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            log_grad_norm=args.log_grad_norm
        )

        # Create storage
        storage = RolloutStorage(
            num_steps=num_steps,
            num_processes=args.num_processes,
            observation_space=observation_space,
            action_space=action_space,
            recurrent_hidden_state_size=args.recurrent_hidden_size,
            recurrent_arch=args.recurrent_arch
        )

        agent = ACAgent(algo=algo, storage=storage).to(device)

    else:
        raise ValueError(f'Unsupported RL algorithm {algo}.')

    return agent
