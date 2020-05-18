set -e

echo "Try to create an virtual environnement... with pip"

python3 -m virtualenv covid-venv


echo "upgrade pip"

pip install --upgrade pip

echo "upgrading pip done"


echo "*********************************************************"
echo "Successfully created the virtual environment!"
echo "It is located at: $(pwd)/covid-venv"
echo "In order to activate this vitualenv:"
echo "$ source activate.sh --"
echo "*********************************************************"
echo ""