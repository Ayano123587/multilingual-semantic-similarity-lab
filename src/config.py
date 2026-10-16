from pathlib import Path

DATA = Path('data')
LANGUAGES = ('amh', 'hau', 'kin')
DATA_REVISION = '3c5941ddffdc9d833a5528f49dd13ed9087c0dba'
DATA_REPOSITORY = 'semantic-textual-relatedness/Semantic_Relatedness_SemEval2024'
MODEL_ID = 'Davlan/afro-xlmr-base'
MODEL_REVISION = '25f27299c247a6b73a767bd82d12444138b19337'
SOURCE_HASHES = {
    'amh_train.csv':'33847fcf914f96e1e47b2806362958afd1be365bbd464a51a276d6782aac96f6',
    'amh_dev_with_labels.csv':'3cf303cd409b0e9eb0ac67d1c8b0c559da57a484b37c370f55248bb8f4df5515',
    'amh_test_with_labels.csv':'8b88797a6f3b22608f3159d6ca47bcf46fb012d248060bbbe1e8f66245f7b422',
    'hau_train.csv':'fef1e27aa6da2295c9c5ce64c8e93ea018070cc6d629aa1d998572d917771618',
    'hau_dev_with_labels.csv':'077a10d9710ad3a4f241cfd41a0b404966d6586811f25f192e3db699985472e3',
    'hau_test_with_labels.csv':'7f166eab81b377e4856d96d36a52e58a862fe4c075acfd4f0c21f6153e5c4f3c',
    'kin_train.csv':'03cb184d2bf4bfd91267f555a794414773854bbb74da409a4bdabc0c1505c2f3',
    'kin_dev_with_labels.csv':'b0bf3b918ecfacfc837d1a909a9ad1f978c185a40eb8e6f4e43887c707d3da71',
    'kin_test_with_labels.csv':'a9258444caea42153f159955e075070f191b8481e939c13c31f074fd47279282'}
