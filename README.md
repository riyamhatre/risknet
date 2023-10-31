Put all your good documentation here. 

To update the version:
- `rm -rf dist build` to remove build folder
- manually remove "egg-info" folder
- update setup.cfg's version number depending on if major, minor, or bug change
- rerun `python3 -m build` (you should get a new dist folder + egg folder in \src)
   - THIS SHOULD CREATE A NEW binary file where version is UPDATED
- rerun `python3 -m twine upload --repository testpypi dist/*`
   - Username: __token__
   - Password: [testpypi password]
   - If you did NOT update the version # before running `build` then you will get an error :/