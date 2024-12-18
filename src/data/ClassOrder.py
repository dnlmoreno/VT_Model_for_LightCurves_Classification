class ClassOrder:
    datasets_order_classes = {
        "elasticc_1": [
            'CART', 'Iax', '91bg', 'Ia', 'Ib/c', 'II', 'SN-like/Other', 'SLSN', 
            'PISN', 'TDE', 'ILOT', 'KN', 'M-dwarf Flare', 'uLens', 'Dwarf Novae', 
            'AGN', 'Delta Scuti', 'RR Lyrae', 'Cepheid', 'EB',
        ],
        "macho": [
            "RRab", "RRc", "Cep_0", "Cep_1", "EC",
        ],
        "macho_multiband": [
            "RRab", "RRc", "Cep_0", "Cep_1", "EC",
        ],
    }

    @staticmethod
    def get_order(name_dataset):
        return ClassOrder.datasets_order_classes.get(name_dataset, [])