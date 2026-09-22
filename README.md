# FlowDAgger

Reference implementation for the paper *FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space*.

Project page: https://microsoft.github.io/FlowDAgger

FlowDAgger is latent-space DAgger for flow-matching and diffusion based robot policies. Instead of
fine-tuning the base policy, it learns a small steering network that outputs an
observation-conditioned initial latent for the policy's sampler. Expert
corrections are mapped back into the sampler's initial-latent space by inverting
the policy's sampling ODE, and the steering network is trained to match those
inverted targets with a behavior-cloning loss.

This repo contains minimal reference implementations for two base-policy
backends. Each backend has its own dependencies and runnable example.

## How this example works

1. Roll out the base policy. A steering network outputs an
   observation-conditioned initial latent for the sampler.
2. An intervention handler hands control to a scripted expert.
3. Each expert action chunk is inverted through the policy's sampler to estimate
   the initial latent that would have produced it.
4. The steering network is trained to match those inverted latent targets (MSE).
5. Repeat. Over time the steering network reproduces expert behavior without
   touching the base-policy weights.

## Layout

```
shared/             scripted expert, task registry, intervention handler
flowdagger_pi05/    JAX, pi0.5 base, MetaWorld assembly
                    (openpi is a git submodule under flowdagger_pi05/openpi)
flowdagger_gr00t/   PyTorch, GR00T N1.7 base, LIBERO-90 task 57
```

## Getting started

The backends use separate Python environments because their JAX and PyTorch
stacks have different dependencies. Follow the README for the backend you want
to run:

- [pi0.5 on MetaWorld assembly](flowdagger_pi05/README.md)
- [GR00T N1.7 on LIBERO-90 task 57](flowdagger_gr00t/README.md)

The pi0.5 backend uses openpi as a git submodule:

```
git submodule update --init flowdagger_pi05/openpi
```

## Citation

```bibtex
@inproceedings{murray2026flowdagger,
  title={{FlowDAgger}: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space},
  author={Murray, Michael and Chen, Daphne and Bagaria, Simran and Fortier, Dean and Hellebrekers, Tess and Mullins, Galen and Gajarla, Harshavardhan and Mees, Oier and Cakmak, Maya and Kolobov, Andrey},
  booktitle={Proceedings of the 10th Conference on Robot Learning},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
