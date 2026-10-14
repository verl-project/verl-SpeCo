// Only this trusted default-branch controller handles comments and API writes.
const REQUEST = '<!-- speco-ci-request ';
const CLAIM = '<!-- speco-ci-dispatch ';
const bot = comment => comment.user?.login === 'github-actions[bot]' && comment.user?.type === 'Bot';

function parseCommand(body) {
  const match = body.trim().match(/^\/ci\s+(run|approve)\s+([a-z0-9-]+)$/);
  if (!match) return null;
  if (match[1] === 'approve' && !/^[1-9][0-9]*$/.test(match[2])) return null;
  return {action: match[1], argument: match[2]};
}

function metadata(comment, prefix) {
  if (!bot(comment) || !comment.body.startsWith(prefix)) return null;
  try { return JSON.parse(comment.body.slice(prefix.length).split(' -->')[0]); }
  catch { return null; }
}

async function handle({github, context, cases, settings = {}}) {
  const event = context.payload;
  if (!event.issue?.pull_request || event.comment.user.type === 'Bot') return;
  const command = parseCommand(event.comment.body);
  if (!command) return;
  const scope = {...context.repo, issue_number: event.issue.number};
  const reply = body => github.rest.issues.createComment({...scope, body});
  const {data: pr} = await github.rest.pulls.get({...context.repo, pull_number: scope.issue_number});
  if (pr.state !== 'open' || !pr.head.repo) return reply('CI: PR must be open with an available head repository.');
  const {data: permission} = await github.rest.repos.getCollaboratorPermissionLevel({
    ...context.repo, username: event.comment.user.login,
  });
  const maintainer = ['admin', 'maintain', 'write'].includes(permission.permission);
  if (command.action === 'run') {
    if (event.comment.user.id !== pr.user.id && !maintainer) return reply('CI: only the PR author or a maintainer may request tests.');
    const example = cases[command.argument];
    if (!example) return reply('CI: unknown case. Available cases: ' + Object.keys(cases).map(x => '`' + x + '`').join(', '));
    const record = {request: event.comment.id, body: event.comment.body, updated: event.comment.updated_at,
      sha: pr.head.sha, repository: pr.head.repo.full_name, alias: command.argument, example};
    return reply(REQUEST + JSON.stringify(record) + ' -->\n' +
      `Requested **${record.alias}** at \`${record.sha}\`. A maintainer can approve with \`/ci approve ${record.request}\`. New commits require a new request.`);
  }
  if (!maintainer) return reply('CI: approval requires repository write/maintain/admin permission.');
  const id = Number(command.argument);
  if (!Number.isSafeInteger(id)) return reply('CI: invalid request comment ID.');
  const comments = await github.paginate(github.rest.issues.listComments, {...scope, per_page: 100});
  const original = comments.find(c => c.id === id);
  const record = comments.map(c => metadata(c, REQUEST)).find(r => r?.request === id);
  if (!original || !record || original.body !== record.body || original.updated_at !== record.updated) {
    return reply('CI: request missing or edited; post a new /ci run command.');
  }
  if (comments.some(c => metadata(c, CLAIM)?.request === id)) return reply('CI: dispatch already attempted for this request. Post a new request to retry.');
  if (record.sha !== pr.head.sha || record.repository !== pr.head.repo.full_name || cases[record.alias] !== record.example) {
    return reply('CI: PR head or case changed; post a new /ci run command.');
  }
  const config = settings[record.alias] || {};
  const inputs = {pr_number: String(scope.issue_number), expected_sha: record.sha, example: record.example,
    environment_json: JSON.stringify(config.environment || {}), hydra_args: config.hydra_args || '',
    verl_ref: config.verl_ref || 'release/v0.8.0'};
  if (config.npu_image) inputs.npu_image = config.npu_image;
  // Claim before dispatch: reruns and duplicate approval comments fail closed.
  const claim = await reply(CLAIM + JSON.stringify({request: id}) + ' -->\nDispatching approved case at `' + record.sha + '`.');
  try {
    await github.rest.actions.createWorkflowDispatch({...context.repo, workflow_id: 'examples_e2e_manual.yml',
      ref: context.payload.repository.default_branch, inputs});
    await github.rest.issues.updateComment({...context.repo, comment_id: claim.data.id,
      body: CLAIM + JSON.stringify({request: id}) + ' -->\nDispatched **' + record.alias + '** at `' + record.sha + '`. Results will appear on this commit.'});
  } catch (error) {
    await github.rest.issues.updateComment({...context.repo, comment_id: claim.data.id,
      body: CLAIM + JSON.stringify({request: id}) + ' -->\nDispatch failed. Post a new request to retry.'});
    throw error;
  }
}
module.exports = {handle, parseCommand, metadata};
