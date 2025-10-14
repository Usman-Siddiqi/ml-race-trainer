from race_environment import RaceCarEnv

if __name__ == '__main__':
    env = RaceCarEnv(render_mode=None)
    obs, info = env.reset()
    print('total_checkpoints =', env.total_checkpoints)
    print('start_position =', env.start_position, 'start_angle =', env.start_angle)
    print('\nIndex : (x,y)')
    for i, (x, y) in enumerate(env.checkpoints):
        marker = '<-- current' if i == env.current_checkpoint else ''
        print(f"{i:2d}: ({x:.1f}, {y:.1f}) {marker}")
    print('\ncurrent_checkpoint index =', env.current_checkpoint)
    print('last_distance_to_checkpoint =', env.last_distance_to_checkpoint)
    print('checkpoint_progress (this lap) =', env.checkpoints_passed_this_lap, '/', env.total_checkpoints)
