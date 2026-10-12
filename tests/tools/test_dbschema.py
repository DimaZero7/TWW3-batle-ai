"""tools.nn.dbschema: the RPFM schema's RON read down to our tables, and its copy config/db_schema.json."""
import json

import pytest

from tools.nn import dbschema

RON = '''(
    version: 5,
    definitions: {
        "a_tables": [
            (
                version: 0,
                fields: [
                    (
                        name: "old",
                        field_type: I32,
                        is_key: false,
                        default_value: Some("0"),
                        is_reference: Some(("other_tables", "key")),
                        description: "a \\"quoted\\" text",
                        enum_values: {},
                    ),
                ],
                localised_fields: [],
            ),
            (
                version: 2,
                fields: [
                    (name: "key", field_type: StringU8, is_key: true, default_value: None),
                    (name: "flag", field_type: Boolean, is_key: false, default_value: Some("false")),
                    (name: "maybe", field_type: OptionalStringU8, is_key: false, default_value: None),
                    (name: "x", field_type: F32, is_key: false, default_value: Some("1.5")),
                ],
                localised_fields: [],
            ),
        ],
        "b_tables": [
            (version: 1, fields: [(name: "n", field_type: SequenceU32(()), is_key: false)]),
        ],
    },
    patches: {
        "a_tables": {"key": {"not_empty": "true"}},
    },
)'''


def test_ron_tables_read_every_version_in_written_order():
    got = dbschema.ron_tables(RON, ["a"])
    assert got == {"a": {"0": [["old", "I32"]],
                         "2": [["key", "StringU8"], ["flag", "Boolean"], ["maybe", "OptionalStringU8"], ["x", "F32"]]}}
    assert dbschema.ron_tables(RON, ["a"], versions={"a": {2}})["a"].keys() == {"2"}


def test_ron_tables_refuse_what_they_cannot_read():
    with pytest.raises(ValueError, match="field type"):
        dbschema.ron_tables(RON, ["b"])
    with pytest.raises(KeyError, match="c_tables"):
        dbschema.ron_tables(RON, ["c"])
    with pytest.raises(ValueError, match="newer schema"):
        dbschema.ron_tables(RON, ["a"], versions={"a": {3}})


def test_extract_without_the_ron_says_how_to_download_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="curl -L -o .* " + dbschema.URL):
        dbschema.extract(tmp_path / "schema_wh3.ron")


def test_the_copy_has_one_version_of_every_table_we_read():
    doc = dbschema.load()
    assert set(doc["tables"]) == set(dbschema.TABLES)
    assert doc["_source"]["sha256"] and doc["_source"]["url"] == dbschema.URL
    for table, versions in doc["tables"].items():
        assert len(versions) == 1, table
        for fields in versions.values():
            names = [n for n, _ in fields]
            assert len(names) == len(set(names)), table
            assert all(t in dbschema.TYPES for _, t in fields), table


def test_fields_name_a_missing_version_or_table():
    assert dbschema.fields("_kv_morale", None) == [("key", "StringU8"), ("value", "F32")]
    with pytest.raises(ValueError, match="refresh"):
        dbschema.fields("_kv_morale", 7)
    with pytest.raises(KeyError):
        dbschema.fields("not_a_table", 0)


def test_dumps_writes_one_field_a_line_and_reads_back():
    doc = {"_description": "d", "_source": {"sha256": "x"}, "tables": {"a": {"2": [["key", "StringU8"], ["x", "F32"]]}}}
    text = dbschema.dumps(doc)
    assert json.loads(text) == doc and '    ["x", "F32"]\n' in text


@pytest.mark.skipif(not dbschema.RON.exists(), reason="no RPFM schema in build/rpfm (not in Git)")
def test_the_copy_is_the_ron_cut_down():
    doc = dbschema.load()
    versions = {t: {int(v) for v in vs} for t, vs in doc["tables"].items()}
    fresh = dbschema.extract(dbschema.RON, versions)
    if fresh["_source"]["sha256"] != doc["_source"]["sha256"]:
        pytest.skip("build/rpfm holds another schema download")
    assert dbschema.dumps(fresh) == dbschema.OUT.read_text(encoding="utf-8")
