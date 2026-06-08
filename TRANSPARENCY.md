# Microsoft Responsible AI Transparency Documentation for Research


# FlowDAgger

## Overview

FlowDAgger is a research code release for latent-space DAgger adaptation of
generative robot policies. It adapts a frozen flow-matching or diffusion robot
policy by learning a small steering network that predicts the noise input to the
policy sampler, rather than fine-tuning the base policy weights. Expert
corrections are mapped back into the base policy's noise space by inverting the
policy's sampling process, and the steering network is trained on those
noise-space targets with a behavior-cloning loss.

This repository is a minimal, self-contained reference implementation for the
pi0.5 base policy on the MetaWorld assembly task. The broader FlowDAgger paper
describes additional simulation, real-robot, and base-policy experiments beyond
the scope of this code release.

Related assets:

- Paper/project website: https://flowdagger.github.io

## What Can FlowDAgger Do

FlowDAgger was developed to study rapid, human-in-the-loop adaptation of frozen
generative robot policies. In this reference implementation, it can:

- Run the pi0.5 MetaWorld base policy on MetaWorld V3 assembly.
- Collect intervention data from a scripted MetaWorld expert when the policy
  stalls, diverges, or reaches a scheduled takeover point.
- Invert each expert action chunk through the pi0.5 flow-matching sampler to
  recover the noise that would have generated the expert action.
- Train a lightweight steering policy to predict those noise targets from
  observations.
- Evaluate the adapted policy and write evaluation videos, JSONL results, and
  steering checkpoints.

A detailed discussion of FlowDAgger, including how the method was developed and
tested, can be found in the paper/project materials at:
https://flowdagger.github.io

## Intended Uses

FlowDAgger is best suited for research on robot policy adaptation, imitation
learning, online learning from interventions, latent-space control, and
evaluation of generative robot policies in simulation. This release is intended
to facilitate reproduction of the reference pi0.5/MetaWorld assembly result and
to provide a concrete implementation for researchers who want to build related
adaptation methods.

FlowDAgger is being shared with the research community to facilitate
reproduction of our results and foster further research in this area.

FlowDAgger is intended to be used by domain experts who are independently
capable of evaluating robot policy behavior, inspecting generated trajectories,
and deciding whether outputs are safe or appropriate before acting on them.

## Out-of-Scope Uses

FlowDAgger is not well suited for direct deployment in commercial, consumer, or
unsupervised real-world robotics applications. This code release has been
implemented and documented as a research reference, and the repository's
shipping task configuration covers MetaWorld assembly with a scripted expert.

We do not recommend using FlowDAgger in commercial or real-world applications
without further testing and development. It is being released for research
purposes.

FlowDAgger was not designed or evaluated for all possible downstream purposes.
Developers should consider its inherent limitations as they select use cases,
and evaluate and mitigate for accuracy, safety, and fairness concerns specific
to each intended downstream use.

Without further testing and development, FlowDAgger should not be used in
sensitive domains where inaccurate outputs could suggest actions that lead to
injury or negatively impact an individual's legal, financial, or life
opportunities.

We do not recommend using FlowDAgger in the context of high-risk decision making
such as law enforcement, legal, finance, or healthcare.

## How to Get Started

From the repository root:

```bash
git submodule update --init flowdagger_pi05/openpi
python -m venv .venv
source .venv/bin/activate
pip install -e flowdagger_pi05/openpi
pip install -r flowdagger_pi05/requirements.txt
```

Run the reference experiment:

```bash
cd flowdagger_pi05
python train_flowdagger.py --env metaworld --task_key metaworld_assembly --seed 42
```

On a headless machine, set MuJoCo to render off-screen:

```bash
MUJOCO_GL=egl python train_flowdagger.py --env metaworld --task_key metaworld_assembly --seed 42
```

The pi0.5 MetaWorld checkpoint is fetched automatically from Hugging Face on
the first run and cached under `~/.cache/huggingface`. To use a local
checkpoint instead, set `METAWORLD_CHECKPOINT` to a directory containing
`params/` and `assets/`.

WandB logging is opt-in. Pass `--prefix <name>` to create a run. Without
`--prefix`, no WandB run is created. Outputs are written to `$EXP/<run-name>` if
`EXP` is set, otherwise to `~/flowdagger_runs/<run-name>`.

## Evaluation

FlowDAgger was evaluated on its ability to improve the success rate of frozen
generative robot policies after a limited number of additional rollouts and
expert interventions.

A detailed discussion of our evaluation methods and results can be found in the
paper/project materials at: https://flowdagger.github.io

## Evaluation Methods

We used task success rate to measure FlowDAgger's performance. The paper also
reports sample efficiency over adaptation rollouts, comparisons against
alternative adaptation techniques, preservation of the base policy's behavior on
held-out tasks, and training memory footprint.

In the paper, FlowDAgger is compared against the frozen base policy, supervised
fine-tuning, LoRA-DAgger, Residual-DAgger, and DSRL. Simulation experiments use
MetaWorld tasks, and the broader paper includes real-hardware manipulation
tasks. The model used for this code release is pi0.5 with the
`pi05-metaworld` checkpoint.

Results may vary if FlowDAgger is used with a different base model, checkpoint,
robot platform, task, intervention policy, expert, hyperparameter setting, or
training/evaluation protocol.

No separate systematic Responsible AI or Defense Security Board testing is
included in this repository. Users should perform their own safety, security,
privacy, and misuse evaluations before extending the code to new settings.

## Evaluation Results

At a high level, the paper finds that FlowDAgger improves frozen generative
robot policies with a small number of corrections, outperforms several
adaptation baselines, and better preserves pretrained behaviors on held-out
tasks than weight-space fine-tuning baselines.

For the repository's reference pi0.5/MetaWorld assembly recipe, the README
reports that seed 42 evaluation improves from the pi0.5 base policy success
rate of about 0.5 to about 0.9 by the default `--max_steps 2000` setting, with a
peak of about 0.96 over 25 evaluation rollouts. Exact results can vary with
hardware, dependencies, random seeds, simulator behavior, and checkpoint
versions.

## Limitations

FlowDAgger was developed for research and experimental purposes. Further
testing and validation are needed before considering its application in
commercial or real-world scenarios.

FlowDAgger adapts a base policy through its noise input. Corrections invert
faithfully only within a bounded neighborhood of the base policy's action
manifold. If a correction is too far from what the base policy can express, the
inversion may recover only the nearest reachable action.

Inversion accuracy can degrade for base policies with stiff or poorly
conditioned velocity fields, or when a correction falls between modes of a
sharply multi-modal action distribution.

As with any DAgger-style method, FlowDAgger is only as good as the corrections
it receives on the states where they are provided. Poor, inconsistent, delayed,
or unsafe corrections can degrade the adapted policy.

This code release ships a minimal pi0.5/MetaWorld assembly implementation. It
does not include all experiments, robots, base-policy families, or datasets
described in the broader paper.

FlowDAgger was designed and tested primarily in robotics settings where task
prompts and documentation use English. Performance with other languages,
prompts, or interfaces may vary and should be assessed by qualified users.

Outputs generated by AI systems, including robot-policy actions or trajectories,
may include errors, unexpected behavior, or speculation-like responses in
language-conditioned components. Users are responsible for assessing the
accuracy and safety of generated behavior. All decisions leveraging outputs of
the system should be made with human oversight and not be based solely on system
outputs.

FlowDAgger inherits any biases, errors, or omissions produced by its base model.
Developers are advised to choose an appropriate base model carefully, depending
on the intended use case.

This release uses pi0.5 through the OpenPI implementation and the
`pi05-metaworld` checkpoint. See the checkpoint and upstream OpenPI
documentation for additional capabilities and limitations.

FlowDAgger inherits any biases, errors, or omissions characteristic of its
training data, base checkpoint, scripted experts, simulator, and task
definitions. These issues may be amplified by downstream adaptation.

There has not been a systematic effort to ensure that systems using FlowDAgger
are protected from security vulnerabilities such as indirect prompt injection
attacks. Any systems using it should take proactive measures to harden their
systems as appropriate, particularly if adding language, web, teleoperation,
logging, or remote-control interfaces.

## Best Practices

Better performance can be achieved by using a base policy whose pretrained
behavior is already close to the target task, providing high-quality and timely
expert corrections, validating inversion error, evaluating over multiple random
seeds and held-out rollouts, and monitoring whether adaptation harms skills that
the base policy already performs well.

Users should keep a human supervisor in the loop when collecting corrections or
testing adapted policies, especially when moving beyond simulation. Real-world
robotics use should include appropriate physical safety systems, emergency stop
procedures, workspace isolation, speed and force limits, and task-specific risk
assessment.

Users are responsible for sourcing their datasets legally and ethically. This
could include securing appropriate rights, ensuring consent for use of
audio/images, and/or anonymizing data prior to use in research.

Users are reminded to be mindful of data privacy concerns and are encouraged to
review the privacy policies associated with any models, datasets, logging
services, and data storage solutions interfacing with FlowDAgger.

It is the user's responsibility to ensure that use of FlowDAgger complies with
relevant data protection regulations and organizational guidelines.

Developers should follow transparency best practices and inform end-users when
they are interacting with an AI-enabled robot or AI system.

We strongly encourage users to use LLMs and multimodal models that support
robust Responsible AI mitigations, such as Azure OpenAI services, when extending
this code with foundation-model interfaces. Such services continually update
their safety and Responsible AI mitigations with current industry standards for
responsible use. For more on Azure OpenAI best practices when employing
foundation models for scripts and applications, see:

- What is Azure AI Content Safety?
  https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview
- Overview of Responsible AI practices for Azure OpenAI models
  https://learn.microsoft.com/en-us/legal/cognitive-services/openai/overview
- Azure OpenAI Transparency Note
  https://learn.microsoft.com/en-us/legal/cognitive-services/openai/transparency-note
- OpenAI Usage Policies
  https://openai.com/policies/usage-policies
- Azure OpenAI Code of Conduct
  https://learn.microsoft.com/en-us/legal/cognitive-services/openai/code-of-conduct

## License

MIT License.

Nothing disclosed here, including the Out-of-Scope Uses section, should be
interpreted as or deemed a restriction or modification to the license the code is
released under.

## Trademarks

This project may contain trademarks or logos for projects, products, or
services. Authorized use of Microsoft trademarks or logos is subject to and must
follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or
logos in modified versions of this project must not cause confusion or imply
Microsoft sponsorship.

Any use of third-party trademarks or logos are subject to those third party's
policies.

## Contact

This research was conducted by members of [Microsoft Research](https://www.microsoft.com/en-us/research/). We welcome feedback and collaboration from our audience. If you have
suggestions, questions, or observe unexpected or offensive behavior in this
technology, please contact the release maintainers.

If the team receives reports of undesired behavior or identifies issues
independently, we will update this repository with appropriate mitigations.
