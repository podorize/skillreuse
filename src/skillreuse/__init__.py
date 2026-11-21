"""SkillReuse: does the text of an agent skill predict whether other developers copy it?

The package follows the phases of the research protocol (paper/protocol.md):

    download.py   Download of the GitSkills dataset (fixed revision, checked by SHA-256)
    collect.py    Phase 1: sample of skills and dating of their copies through the GitHub API
    features.py   Text attributes extracted from a SKILL.md file
    cohort.py     Phase 2: T0, author, time to first copy, attributes and author split
    study.py      Phases 3-7: exploratory analysis, model selection, final models,
                  single holdout evaluation, robustness checks and interpretation
    stats.py      Statistics used by study.py (Holm, Nadeau-Bengio, author bootstrap)
    report.py     Figures of the paper
    cli.py        Command line entry point: ``skillreuse <command>``
"""

__version__ = "0.1.0"
