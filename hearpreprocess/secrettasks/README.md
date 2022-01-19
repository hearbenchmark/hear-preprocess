# hear2021-secret-tasks

Luigi pipelines to create HEAR 2021 secret tasks.

This is a submodule in hearpreprocess, not a standalone module..

The commands for running the pipeline remains the same,
and secret tasks are added. Besides individual secret tasks:
```
python3 -m heareval.tasks.runner all-secret --mode all
```

## Development

Make sure you have pre-commit hooks installed:
```
pre-commit install
```
