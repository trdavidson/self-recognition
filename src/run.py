import time
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from protocols import ResponseOnlyProtocol, QuestionGenProtocol, VerdictOnlyProtocol
from utils import unpack_nested_yaml, printv, get_inference_root_overrides


def questions_only(cfg, output_dir, debug):
    print('run questions_only protocol')

    initialization_prompt = cfg.experiments.initialization_prompt_path

    judge = instantiate(cfg.experiments.judge, initialization_prompt_path=initialization_prompt,
                        generate_new=True, teacher_forcing=False)
    temperature = cfg.experiments.temperature_override
    if temperature >= 0:
        judge.model.temperature = temperature
    protocol_params = cfg.experiments.protocol
    protocol = QuestionGenProtocol(judge=judge, output_dir=output_dir,
                                   **protocol_params)
    protocol.run()


def responses_only(cfg, output_dir, debug):
    printv('run responses_only protocol...', v=1, c='yellow')
    contestants = [instantiate(k) for k in cfg.experiments.contestant_models]
    temperature = cfg.experiments.temperature_override
    if temperature >= 0:
        for c in contestants:
            c.model.temperature = temperature

    protocol_params = cfg.experiments.protocol
    for k, v in protocol_params.items():
        printv(f'  {k}:'.ljust(30) + f'{v}', c='yellow', v=1)
    print('')

    protocol = ResponseOnlyProtocol(contestants=contestants, debug=debug, output_dir=output_dir,
                                    **protocol_params)
    protocol.run()


def verdict_only(cfg, output_dir, debug):
    printv(f'run verdict_only protocol: {cfg.experiments.judge.model.model_name}..', c='yellow', v=1)
    judge = instantiate(cfg.experiments.judge)
    protocol_params = cfg.experiments.protocol

    temperature = cfg.experiments.temperature_override
    if temperature >= 0:
        judge.model.temperature = temperature
    for k, v in protocol_params.items():
        printv(f'  {k}:'.ljust(30) + f'{v}', c='yellow', v=1)
    print('')

    protocol = VerdictOnlyProtocol(judge, debug=debug, output_dir=output_dir,
                                   **protocol_params)
    protocol.run()


@hydra.main(version_base=None, config_path="configs", config_name="inference_root")
def main(cfg: DictConfig):
    """
    Main function to run the protocol
    1. generate_questions: python src/run.py defaults.experiments=generate_questions
    2. generate_responses: python src/run.py defaults.experiments=generate_responses protocol
    3. generate_verdicts:  python src/run.py defaults.experiments=generate_verdicts protocol

    Modify the relevant config files in the src/configs/experiments directory to change the protocol parameters
    """
    with open_dict(cfg['experiments']):
        # unpack nested yaml files
        _ = unpack_nested_yaml(cfg['experiments'])
        # check if any keys are missing and update default run-time overrides
        _ = get_inference_root_overrides(cfg)
        _ = unpack_nested_yaml(cfg['experiments'])

    output_dir = cfg.output_dir
    debug = cfg.debug_mode
    protocol_type = cfg.experiments.protocol_type

    if cfg.sleep_time > 0:
        hours = cfg.sleep_time / 3600
        print(f"[sleep] Going to sleep for {hours: .2f} hours...")
        time.sleep(cfg.sleep_time)  # Sleep for 3600 seconds (1 hour)
        print(f"[wake] Woke up after {hours: .2f} hours!")

    if protocol_type == 'generate_questions':
        questions_only(cfg, output_dir=output_dir, debug=debug)
    elif protocol_type == 'generate_responses':
        responses_only(cfg, output_dir=output_dir, debug=debug)
    elif protocol_type == 'generate_verdicts':
        verdict_only(cfg, output_dir=output_dir, debug=debug)

    else:
        raise NotImplementedError('not supported')


if __name__ == '__main__':
    main()
