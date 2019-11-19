# Remove old distirbution files (committed to git anyway)
rm dist/*

# Make new distribution files
python3 setup.py sdist bdist_wheel

# Upload new distribution files to PyPI
python3 -m twine upload dist/*