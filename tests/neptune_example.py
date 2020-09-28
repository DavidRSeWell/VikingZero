import neptune

# The init() function called this way assumes that
# NEPTUNE_API_TOKEN environment variable is defined.
NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYThhM2FkNmItNGJmOC00NTc3LWI1ZDctNmE1NTYxMDBmMzkyIn0="
neptune.init('befeltingu/sandbox',api_token=NEPTUNE_API_TOKEN)
neptune.create_experiment(name='minimal_example')

# log some metrics

for i in range(100):
    neptune.log_metric('loss', 0.95**i)

neptune.log_metric('AUC', 0.96)
