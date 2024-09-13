import collections
import datetime
import os

try:
    from github import Github
except ImportError:
    raise ImportError('Install PyGithub from https://github.com/PyGithub/PyGithub or via pip')
API_TOKEN = os.environ.get('GITHUB_API_TOKEN', None)
if API_TOKEN is None:
    raise ValueError('Need to specify an API token')

p = Github(API_TOKEN)
last_release = datetime.datetime(year=2023, month=7, day=7)
authors = []
comments = p.get_repo('brian-team/brian2').get_issues_comments(since=last_release)
comment_counter = 0
for comment in comments:
    name = comment.user.name
    if name is None:
        authors.append('`@{login} <https://github.com/{login}>`_'.format(login=comment.user.login))
    else:
        authors.append(
            '{name} (`@{login} <https://github.com/{login}>`_)'.format(
                login=comment.user.login,
                name=name))
    comment_counter += 1
print('Counted {} comments'.format(comment_counter))

issues = p.get_repo('brian-team/brian2').get_issues(since=last_release)
issue_counter = 0
for issue in issues:
    name = issue.user.name
    if name is None:
        authors.append('`@{login} <https://github.com/{login}>`_'.format(login=issue.user.login))
    else:
        authors.append(
            '{name} (`@{login} <https://github.com/{login}>`_)'.format(
                login=issue.user.login,
                name=name))
    issue_counter += 1
print('Counted {} issues'.format(issue_counter))

counted = collections.Counter(authors)
sorted = sorted(counted.items(), key=lambda item: item[1], reverse=True)
for name, contributions in sorted:
    print('{:>4} {}'.format(contributions, name))
