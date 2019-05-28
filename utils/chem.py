"""
RDKit util functions.
"""
import rdkit.Chem as rkc
import deepsmiles as ds


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)

    import rdkit.rdBase as rkrb
    rkrb.DisableLog('rdApp.error')


disable_rdkit_logging()


def read_smi_file(file_path, ignore_invalid=True, num=-1):
    """
    Reads a SMILES file.
    :param file_path: Path to a SMILES file.
    :param ignore_invalid: Ignores invalid lines (empty lines)
    :param num: Parse up to num SMILES.
    :return: A list with all the SMILES.
    """
    return map(lambda fields: fields[0], read_csv_file(file_path, ignore_invalid, num))


def read_csv_file(file_path, ignore_invalid=True, num=-1):
    """
    Reads a SMILES file.
    :param file_path: Path to a CSV file.
    :param ignore_invalid: Ignores invalid lines (empty lines)
    :param num: Parse up to num rows.
    :return: An iterator with the rows.
    """
    with open(file_path, "r") as csv_file:
        for i, row in enumerate(csv_file):
            if i == num:
                break
            fields = row.rstrip().split(",")
            if fields:
                yield fields
            elif not ignore_invalid:
                yield None


def to_mol(smi):
    """
    Creates a Mol object from a SMILES string.
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if smi:
        return rkc.MolFromSmiles(smi)


def to_smiles(mol):
    """
    Converts a Mol object into a canonical SMILES string.
    :param mol: Mol object.
    :return: A SMILES string.
    """
    return rkc.MolToSmiles(mol, isomericSmiles=False)


DEEPSMI_CONVERTERS = {
    "rings": ds.Converter(rings=True),
    "branches": ds.Converter(branches=True),
    "both": ds.Converter(rings=True, branches=True)
}


def to_deepsmiles(smi, converter="both"):
    """
    Converts a SMILES strings to the DeepSMILES alternative.
    :param smi: SMILES string.
    :return : A DeepSMILES string.
    """
    return DEEPSMI_CONVERTERS[converter].encode(smi)


def from_deepsmiles(deepsmi, converter="both"):
    """
    Converts a DeepSMILES strings to the SMILES alternative.
    :param smi: DeepSMILES string.
    :return : A SMILES string or None if it's invalid
    """
    try:
        return DEEPSMI_CONVERTERS[converter].decode(deepsmi)
    except:  # pylint:disable=bare-except
        return None
