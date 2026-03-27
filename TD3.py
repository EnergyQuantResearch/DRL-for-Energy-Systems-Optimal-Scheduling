from trainer import build_experiment_config, run_experiment


if __name__ == '__main__':
    run_experiment(build_experiment_config('td3'))
