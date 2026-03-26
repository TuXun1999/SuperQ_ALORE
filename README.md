# Set of the Project
Everything in README(old).md is deployed, except the Omniverse setup. (The IDE setup may not
be 100% successful though, because it seems that isaacsim can be detected by VScode, but isaaclab cannot)

# Deployment of low-level controller
In your terminal, run 
```
python3 ./scripts/zero_agent.py --task Template-Superq-Alore-v0 --num_envs 20
```

Then, you should observe the robot moving forward with a constant speed. 

You can set up the attributes in "./scripts/zero_agent.py". 