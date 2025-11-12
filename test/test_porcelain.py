# -----------------------------------------------------------------------------
# Copyright (C) 2026 Iliyas Jorio.
# This file is part of GitFourchette, distributed under the GNU GPL v3.
# For full terms, see the included LICENSE file.
# -----------------------------------------------------------------------------

"""
Unit tests for gitfourchette.porcelain (Repo, listall_remote_branches, in_gitdir).
"""

from pathlib import Path

import pytest

from .util import unpackRepo, RepoContext, WINDOWS


def testListallRemoteBranchesWithSymbolicRef(tempDir):
    """
    Repo.listall_remote_branches() must not raise when the repo contains
    symbolic refs (e.g. from the "repo" tool: refs/remotes/m/master -> refs/remotes/origin/master).
    Right-clicking such a branch in the sidebar would previously cause a stack trace.
    """
    wd = unpackRepo(tempDir)
    # Create a symbolic ref like repo tool: refs/remotes/<manifest-name>/<branch>
    # pointing at refs/remotes/origin/<branch>. "m" is not a configured remote.
    refs_m_dir = Path(wd.rstrip("/")) / ".git" / "refs" / "remotes" / "m"
    refs_m_dir.mkdir(parents=True, exist_ok=True)
    (refs_m_dir / "master").write_text("ref: refs/remotes/origin/master\n")

    with RepoContext(wd) as repo:
        # Must not raise KeyError or similar
        result = repo.listall_remote_branches()
    assert "origin" in result
    assert "master" in result["origin"]


def testListallRemoteBranchesWithStaleSymbolicRef(tempDir):
    """
    Stale symbolic refs (pointing to a ref that no longer exists) must be
    skipped without raising.
    """
    wd = unpackRepo(tempDir)
    refs_m_dir = Path(wd.rstrip("/")) / ".git" / "refs" / "remotes" / "m"
    refs_m_dir.mkdir(parents=True, exist_ok=True)
    (refs_m_dir / "master").write_text("ref: refs/remotes/origin/nonexistent\n")

    with RepoContext(wd) as repo:
        result = repo.listall_remote_branches()
    # Should still have origin's branches; stale symref is skipped
    assert set(result.keys()) == {"origin"}


@pytest.mark.skipif(WINDOWS, reason="symlinks are flaky on Windows")
def testInGitdirWithSymlinkedRepo(tempDir):
    """
    Repo.in_gitdir() must not raise ValueError when the repo path is a symlink.
    Previously is_relative_to(parent) failed because the resolved path was
    compared against the unresolved (symlink) parent.
    """
    wd = unpackRepo(tempDir)
    real_path = Path(wd.rstrip("/")).resolve()
    link_path = Path(tempDir.name) / "repo-link"
    link_path.symlink_to(real_path)
    repo_path = str(link_path) + "/"

    with RepoContext(repo_path) as repo:
        # Must not raise ValueError("Won't resolve absolute path outside gitdir")
        config_path = repo.in_gitdir("config", common=True)
    assert config_path.endswith("config")
