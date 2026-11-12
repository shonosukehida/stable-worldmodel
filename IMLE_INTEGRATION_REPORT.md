# IMLE integration（2026-11-12）

作業ブランチ：integration/imle-into-multiview。基準：ab3e681。
移植元：6f7ea56。push・最終マージなし。実機接続なし。

IMLE専用package/test/参照fixtureを選択移植。GPCPolicyにaction_policyを追加し、
既存DiffusionPolicyとGPCの正規化・rollout・候補選択は維持した。
optional extra imleとuv.lockを整合させた。

LeWorldModel multiviewが要求するreward APIが基準コミットにはなかったため、
ユーザー承認の既存未コミットreward.pyとtest_latent_goal_reward.pyを内容を変えずコピーした。
元worktreeの変更は保持。reward比較・回帰検証は両側に同じ前提を使用した。

policy / wm / probingテスト：変更前後とも78 passed、9 failed。
9件は既存WorldModelPolicy fixtureの空action_space。推論実装を変更して解消していない。
最終対象テスト一括は42 passed、25 subtests passed。
IMLEテスト、合成学習checkpoint、実JEPAとの接続、学習済みDP互換性はCPUで検証。
実運用IMLE/指定WM、CUDA、実機は未検証。全体のマージ判定は保留。

詳細・変更ファイル一覧・再実行方法は、同時に統合するLeWorldModelの
`docs/IMLE_INTEGRATION_REPORT.md` を参照。
ローカルログ：`/home/hida/workspace/imle-integration/validation/`。
