import pyvo

tap = pyvo.dal.TAPService('https://almascience.nrao.edu/tap')
result = tap.search("SELECT column_name, description, unit, datatype FROM TAP_SCHEMA.columns WHERE table_name = 'ivoa.obscore' ORDER BY column_name")

print(f'Found {len(result)} columns in ivoa.obscore')

with open('alma_tap_columns.txt', 'w') as f:
    f.write(f"ALMA TAP ivoa.obscore - {len(result)} columns\n")
    f.write("=" * 80 + "\n\n")
    for r in result:
        col = r['column_name']
        dtype = str(r['datatype']) if r['datatype'] else ''
        unit = str(r['unit']) if r['unit'] else ''
        desc = str(r['description']) if r['description'] else ''
        f.write(f"{col}\n  Type: {dtype}, Unit: {unit}\n  {desc}\n\n")
        print(f'{col:35} | {dtype:10} | {unit:10}')

print("\nSaved to alma_tap_columns.txt")
