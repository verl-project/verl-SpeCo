// Resolve only an open PR in the current repository; pin its immutable head SHA.
async function resolveTarget({github, context, number, expectedSha}) {
  const value = (number || '').trim();
  if (!value) {
    return {repository: `${context.repo.owner}/${context.repo.repo}`, sha: context.sha};
  }
  if (!/^[1-9][0-9]*$/.test(value) || !Number.isSafeInteger(Number(value))) {
    throw new Error('PR number must be a positive integer');
  }
  const {data: pr} = await github.rest.pulls.get({...context.repo, pull_number: Number(value)});
  if (pr.state !== 'open' || !pr.head.repo || !/^[0-9a-f]{40}$/.test(pr.head.sha)) {
    throw new Error('PR must be open and have an available head repository and SHA');
  }
  if (expectedSha && pr.head.sha !== expectedSha) throw new Error("PR head changed after approval; request a new run");
  return {repository: pr.head.repo.full_name, sha: pr.head.sha};
}
module.exports = {resolveTarget};
