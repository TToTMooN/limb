Create a new VLA policy launch config for YAM bimanual.

Ask the user:
1. Which policy type? (pi0 / diffusion)
2. Server address and port for the policy server
3. Which cameras to use (names and serial numbers)
4. Action chunk size (how many steps to execute per inference)

Then create a new YAML config at configs/yam/yam_<policy>_bimanual.yaml
based on the structure of configs/yam/yam_viser_bimanual.yaml, but replacing
the agent section with the appropriate policy agent:

For π0:
```yaml
agent:
  _target_: limb.agents.policy_learning.async_pi0_agent.AsyncPi0Agent
  server_url: "http://<host>:<port>"
```

For diffusion policy:
```yaml
agent:
  _target_: limb.agents.policy_learning.diffusion_policy_agent.AsyncDiffusionPolicyAgent
  server_url: "ws://<host>:<port>"
```

Read the actual agent files first to get the correct constructor signatures before writing the config.
