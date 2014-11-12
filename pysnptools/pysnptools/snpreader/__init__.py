"""Tools for reading SNP files


"""

# attempt to import wrapped plink parser
WRAPPED_PLINK_PARSER_PRESENT = True
try:
    import pysnptools.pysnptools.snpreader.wrap_plink_parser
except Exception:
    WRAPPED_PLINK_PARSER_PRESENT = False
