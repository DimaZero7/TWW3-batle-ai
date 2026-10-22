"""tools/nn/train/run.py general_config: the options given on the command line go over an --init's saved general rules
(the deployment, the re-planning, the rounds); what is not given stays as saved; an option against a saved size stops
the run (the weights would not fit). torch: the training container."""
import argparse
import dataclasses

import pytest

torch = pytest.importorskip("torch")

from tools.nn.general import config as gconfig                 # noqa: E402
from tools.nn.general import executor                          # noqa: E402
from tools.nn.model import config                              # noqa: E402
from tools.nn.train import checkpoint, run                     # noqa: E402

TINY = config.preset("v2", d=32, layers=2, heads=2, pointer=16, critic_d=32, critic_layers=1, critic_heads=2,
                     sector_d=16)
SAVED = gconfig.GeneralConfig(card=16, reader_hidden=16, council_heads=2, deploy="net")


def args_of(*line):
    return run.parser().parse_args(list(line))


def test_the_parser_records_only_the_options_given():
    a = args_of("--general-deploy", "script", "--general-entropy-kind", "0.02")
    assert "general_deploy" in a.cli_given and "general_entropy_kind" in a.cli_given
    assert "general_replan_s" not in a.cli_given and "general_card" not in a.cli_given
    assert a.general_deploy == "script" and a.general_replan_s == 30.0      # the defaults are still there
    assert args_of().cli_given == []


def test_a_given_rule_goes_over_the_saved_one():
    gcfg, over = run.general_config(args_of("--general-deploy", "script", "--general-replan-s", "20"), SAVED)
    assert gcfg.deploy == "script" and gcfg.replan_s == 20.0
    assert over == {"deploy": ("net", "script"), "replan_s": (30.0, 20.0)}
    assert gcfg.card == SAVED.card and gcfg.council_heads == SAVED.council_heads       # the sizes as saved


def test_the_kinds_inertia_is_a_rule_an_option_goes_over():
    gcfg, over = run.general_config(args_of("--general-kind-stick", "0"), SAVED)
    assert gcfg.kind_stick == 0.0 and over == {"kind_stick": (1.0, 0.0)}
    old = {k: v for k, v in gconfig.to_dict(SAVED).items() if k != "kind_stick"}     # a checkpoint before it: 1
    assert gconfig.of_dict(old).kind_stick == 1.0
    a = args_of()
    assert (a.general_imitate, a.general_imitate_decay, a.general_exec_general) == (0.3, 300, 0.2)


def test_the_imitations_floor_is_saved_and_the_option_given_goes_over_it():
    from tools.nn.general import learn
    cfg = learn.LearnConfig(imitate=0.3, imitate_min=0.15)
    assert args_of().general_imitate_min == 0.15
    assert run.imitate_floor(args_of(), cfg, {}).imitate_min == 0.15                  # nothing saved: the option's
    assert run.imitate_floor(args_of(), cfg, {"general_imitate_min": 0.05}).imitate_min == 0.05     # the saved one
    c = learn.LearnConfig(imitate=0.3, imitate_min=0.15)
    assert run.imitate_floor(args_of("--general-imitate-min", "0.15"), c,
                             {"general_imitate_min": 0.05}).imitate_min == 0.15       # given (even at the default)
    with pytest.raises(ValueError):
        learn.LearnConfig(imitate_min=-0.1)


def test_without_the_option_the_saved_rule_stays():
    saved = dataclasses.replace(SAVED, deploy="script", replan_s=45.0)
    gcfg, over = run.general_config(args_of("--general-entropy-kind", "0.05"), saved)
    assert gcfg == saved and over == {}


def test_an_option_given_at_its_default_still_goes_over_the_saved_value():
    saved = dataclasses.replace(SAVED, deploy="script")
    gcfg, over = run.general_config(args_of("--general-deploy", "net"), saved)       # "net" is the option's default
    assert gcfg.deploy == "net" and over == {"deploy": ("script", "net")}


@pytest.mark.parametrize("line", [("--general-card", "128"), ("--general-places", "8"), ("--general-kinds", "4"),
                                  ("--general-council-layers", "3"), ("--general-center", "0"),
                                  ("--general-target-grid", "2")])
def test_a_size_option_against_the_saved_size_stops_the_run(line):
    with pytest.raises(SystemExit, match=line[0]):
        run.general_config(args_of(*line), SAVED)


def test_a_size_option_equal_to_the_saved_size_is_fine():
    gcfg, over = run.general_config(args_of("--general-card", "16"), SAVED)
    assert gcfg == SAVED and over == {}


def test_a_namespace_without_the_record_takes_the_options_off_their_defaults():
    a = argparse.Namespace(**{k: v for k, v in vars(args_of("--general-deploy", "script")).items() if k != "cli_given"})
    gcfg, over = run.general_config(a, SAVED)
    assert gcfg.deploy == "script" and over == {"deploy": ("net", "script")}


def test_with_general_applies_the_flags_to_a_loaded_general_and_the_next_checkpoint_keeps_them(tmp_path):
    torch.manual_seed(0)
    actor = executor.GeneralActor(TINY, dataclasses.replace(SAVED, rounds=3))
    path = checkpoint.save(tmp_path / "g.pt", actor)
    loaded = checkpoint.load_policy(path)
    assert loaded.gcfg.deploy == "net"
    out, lcfg = run.with_general(loaded, args_of("--general-deploy", "script", "--general-rounds", "2",
                                                 "--general-entropy-kind", "0.03"))
    assert out is loaded and out.gcfg.deploy == "script" and out.general.cfg is out.gcfg
    assert out.general.groups.rounds == 2 and lcfg.entropy_kind == 0.03
    assert checkpoint.general_of(checkpoint.read(checkpoint.save(tmp_path / "h.pt", out))).deploy == "script"
    again, _ = run.with_general(checkpoint.load_policy(path), args_of())            # no flags: as saved
    assert again.gcfg.deploy == "net" and again.general.groups.rounds == 3
    with pytest.raises(SystemExit):
        run.with_general(checkpoint.load_policy(path), args_of("--general-kinds", "4"))


def test_set_rules_refuses_a_shape_change():
    actor = executor.GeneralActor(TINY, SAVED)
    with pytest.raises(ValueError):
        actor.set_rules(dataclasses.replace(SAVED, card=32))
