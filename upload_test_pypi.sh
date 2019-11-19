# Remove old distirbution files (committed to git anyway)
rm dist/*

# Make new distribution files
python3 setup.py sdist bdist_wheel

# Make new distribution files
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*