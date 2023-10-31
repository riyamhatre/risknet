Put all your good documentation here. 

To update the version:
- `rm -rf dist build` to remove build folder
- manually remove "egg-info" folder
- update setup.cfg's version number depending on if major, minor, or bug change
- rerun `python3 -m build` (you should get a new dist folder + egg folder in \src)
   - THIS SHOULD CREATE A NEW binary file where version is UPDATED
- rerun `python3 -m twine upload --repository testpypi dist/*`
   - Username: __token__
   - Password: [testpypi password starting with pypi]
   - If you did NOT update the version # before running `build` then you will get an error


# History:
0.0.1: Ran into problems with installing pytest-cov
0.0.2: Got error:
`ERROR: Could not find a version that satisfies the requirement dask[complete] (from risknet) (from versions: none) ERROR: No matching distribution found for dask[complete]`
0.0.3: Tried moving dask into [options.extras_require] not install_requires. Got error for flake8
0.0.4: Commented out more packages.
Got error: `error: package directory 'lib3/yaml' does not exist` and `metadata-generation-failed`.
0.0.5: Moved types-YAML into options, still get `'lib3/yaml does not exist'` error :/
0.0.6: we're removing YAML as a test
Got error `ERROR: Could not find a version that satisfies the requirement typing (from risknet) (from versions: none) ERROR: No matching distribution found for typing`
0.0.10: reverting to version 0.0.6, checking if it works --success! But still error `No matching distribution found for typing`