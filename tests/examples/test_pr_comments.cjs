const test = require('node:test');
const assert = require('node:assert/strict');
const {handle, parseCommand} = require('../../ci/pr_comments.cjs');

function fixture() {
  const author = {id: 1, login: 'author', type: 'User'};
  const original = {id: 100, body: '/ci run eagle3', updated_at: 'original', user: author};
  const comments = [original];
  const dispatched = [];
  let permission = 'read';
  const pr = {state: 'open', user: author, head: {repo: {full_name: 'fork/repo'}, sha: 'a'.repeat(40)}};
  const context = {repo: {owner: 'base', repo: 'repo'}, payload: {
    issue: {number: 12, pull_request: {}}, comment: original, repository: {default_branch: 'main'},
  }};
  const github = {rest: {
    pulls: {get: async () => ({data: pr})},
    repos: {getCollaboratorPermissionLevel: async () => ({data: {permission}})},
    issues: {
      listComments: () => {},
      createComment: async ({body}) => {
        const comment = {id: 200 + comments.length, body, user: {login: 'github-actions[bot]', type: 'Bot'}};
        comments.push(comment); return {data: comment};
      },
      updateComment: async ({comment_id, body}) => { comments.find(c => c.id === comment_id).body = body; },
    },
    actions: {createWorkflowDispatch: async args => {dispatched.push(args);}},
  }, paginate: async () => comments};
  const run = () => handle({github, context, cases: {eagle3: 'examples/eagle3.sh'}});
  const approve = () => {
    permission = 'write';
    context.payload.comment = {id: 300, body: '/ci approve 100', user: {id: 2, login: 'maintainer', type: 'User'}};
  };
  return {run, approve, context, pr, comments, dispatched, original,
    setPermission: value => {permission = value;}};
}

test('commands are exact and cannot contain extra shell text', () => {
  assert.deepEqual(parseCommand('/ci run eagle3'), {action: 'run', argument: 'eagle3'});
  for (const text of ['/ci run eagle3;echo', '/ci approve abc', '/ci run eagle3\nextra']) assert.equal(parseCommand(text), null);
});

test('author requests; maintainer approves fork SHA; duplicate approval cannot dispatch twice', async () => {
  const f = fixture(); await f.run();
  assert.equal(f.dispatched.length, 0);
  assert.match(f.comments.at(-1).body, /\/ci approve 100/);
  f.approve(); await f.run();
  assert.equal(f.dispatched.length, 1);
  assert.equal(f.dispatched[0].inputs.expected_sha, 'a'.repeat(40));
  assert.equal(f.dispatched[0].ref, 'main');
  await f.run(); assert.equal(f.dispatched.length, 1);
});

test('outsiders cannot request or approve', async () => {
  const f = fixture();
  f.context.payload.comment.user = {id: 3, login: 'outsider', type: 'User'};
  await f.run(); assert.match(f.comments.at(-1).body, /only the PR author/);
  f.approve(); f.setPermission('read'); await f.run();
  assert.equal(f.dispatched.length, 0);
  assert.match(f.comments.at(-1).body, /approval requires/);
});

test('new commits and edited requests invalidate approval', async () => {
  for (const edited of [false, true]) {
    const f = fixture(); await f.run(); f.approve();
    if (edited) f.original.updated_at = 'edited';
    else f.pr.head.sha = 'b'.repeat(40);
    await f.run(); assert.equal(f.dispatched.length, 0);
  }
});

test('unknown cases and non-PR comments never dispatch', async () => {
  const f = fixture(); f.context.payload.comment.body = '/ci run unknown';
  await f.run(); assert.match(f.comments.at(-1).body, /unknown case/);
  delete f.context.payload.issue.pull_request;
  await f.run(); assert.equal(f.dispatched.length, 0);
});
