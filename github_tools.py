from github import Github


def get_last_commit_hash(source_url):
    """Get last commit time from GitHub URL format: https://github.com/owner/repo@branch"""
    assert source_url.startswith('https://github.com/') and '@' in source_url, \
        f"Expected format: https://github.com/owner/repo@branch, got: {source_url}"

    # Parse URL
    path, branch = source_url.replace('https://github.com/', '').rsplit('@', 1)
    owner, repo_name = path.split('/')

    # Get commit info
    g = Github()  # No auth for public repos
    repo = g.get_repo(f"{owner}/{repo_name}")
    branch_obj = repo.get_branch(branch)

    return branch_obj.commit.sha[:8]