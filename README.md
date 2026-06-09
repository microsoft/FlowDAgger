# FlowDAgger

Reference implementation for the paper *FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space*.

Project page: https://microsoft.github.io/FlowDAgger

FlowDAgger is latent-space DAgger for flow-matching and diffusion based robot policies. Instead of
fine-tuning the base policy, it learns a small steering network that predicts
the initial noise fed to the policy's sampler. Expert corrections are mapped
back into that noise space by inverting the policy's sampling ODE, and the
steering network is trained on the inverted targets with a behavior-cloning
loss.

This repo is a minimal, self-contained reference implementation on the pi0.5
base policy (JAX / openpi), running the MetaWorld assembly task.

## How this example works

1. Roll out the base policy. A steering network predicts the sampling noise.
2. An intervention handler hands control to a scripted expert when the policy
   stalls or diverges.
3. Each expert action chunk is inverted through the policy's sampler to recover
   the noise that would have produced it.
4. The steering network is trained to predict those noise targets (MSE).
5. Repeat. Over time the steering network reproduces expert behavior without
   touching the base-policy weights.

## Layout

```
shared/             scripted expert, task registry, intervention handler
flowdagger_pi05/    the experiment: JAX, pi0.5 base, MetaWorld assembly
                    (openpi is a git submodule under flowdagger_pi05/openpi)
```

## Getting started

```
git submodule update --init flowdagger_pi05/openpi
```

Then follow [flowdagger_pi05/README.md](flowdagger_pi05/README.md) for install
and the exact launch command. The pi0.5 checkpoint is fetched from the Hub
automatically on the first run.

## License

MIT. See [LICENSE](LICENSE).
