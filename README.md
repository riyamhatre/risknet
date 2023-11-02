Put all your good documentation here. 

To update the version:
- `rm -rf dist build` to remove build folder
- manually remove "egg-info" folder. This will change `src` to `src/risknet`.
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
0.0.11: what happens when we remove typing (since it's part of stdlib in python >3.5)?
0.0.12: commented out all packages added by EC
0.0.13: Got new error:
`ERROR: Cannot install pandas==1.3.4 and risknet==0.0.12 because these package versions have conflicting dependencies.
The conflict is caused by:
    risknet 0.0.12 depends on numpy
    pandas 1.3.4 depends on numpy>=1.17.3; platform_machine != "aarch64" and platform_machine != "arm64" and python_version < "3.10"`
Solution: will try setting python > 3.10
### SUCCESS! Version 0.0.13 can download from emily's (base)
Caveats:
- Probably can only work because it has all dependencies already installed in the env (it threw an error when I tried to run it in risknet_test)
- Can only import `risknet.utils.label_prep` since `risknet.utils.encoder`, etc. have local imports to different .py files which Python can't read (??)
But Running `>>> import risknet.utils.label_prep as label_prep, >>> label_prep.label_proc(fm_root, data)` works!!

0.0.14: change setup.py to `if __name__ == "__main__: setup()`.

0.0.16: try compiling on base environment (python 3.12, pip 23.2)