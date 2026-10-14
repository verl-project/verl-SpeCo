const assert = require('node:assert/strict');
const test = require('node:test');
const {resolveTarget} = require('../../ci/pr_target.cjs');

const context = {repo: {owner: 'base', repo: 'repo'}, sha: 'a'.repeat(40)};
const api = data => ({rest: {pulls: {get: async args => {
  assert.equal(args.owner, 'base');
  assert.equal(args.repo, 'repo');
  assert.equal(args.pull_number, 12);
  return {data};
}}}});

test('blank PR uses selected branch SHA without API call', async () => {
  assert.deepEqual(await resolveTarget({github: {}, context, number: ''}), {
    repository: 'base/repo', sha: context.sha,
  });
});

test('fork PR pins its head repository and SHA', async () => {
  const github = api({state: 'open', head: {repo: {full_name: 'fork/repo'}, sha: 'b'.repeat(40)}});
  assert.deepEqual(await resolveTarget({github, context, number: '12'}), {
    repository: 'fork/repo', sha: 'b'.repeat(40),
  });
});

test('invalid PR numbers fail before API access', async () => {
  for (const number of ['0', '-1', 'abc', '1;exit', '9007199254740992']) {
    await assert.rejects(resolveTarget({github: {}, context, number}));
  }
});

test('closed PRs, deleted repositories and malformed SHAs are rejected', async () => {
  for (const data of [
    {state: 'closed'},
    {state: 'open', head: {repo: null, sha: 'b'.repeat(40)}},
    {state: 'open', head: {repo: {full_name: 'fork/repo'}, sha: 'not-a-sha'}},
  ]) {
    await assert.rejects(resolveTarget({github: api(data), context, number: '12'}));
  }
});


test('approved SHA cannot silently change at dispatch time', async () => {
  const github = api({state: 'open', head: {repo: {full_name: 'fork/repo'}, sha: 'b'.repeat(40)}});
  await assert.rejects(resolveTarget({github, context, number: '12', expectedSha: 'a'.repeat(40)}));
});
