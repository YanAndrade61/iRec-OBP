source .env/bin/activate
git clone https://github.com/irec-org/irec.git
mv irec irec-master
mv irec-master/irec/ .
python3 -m pip install pip --upgrade
cd irec-master
python3 -m pip install -e .
cd ..
rm -r irec-master