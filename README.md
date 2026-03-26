# sec-cell-
Image Recognition and Cell Division Analysis of Zebrafish Embryonic Epidermal Cells

斑馬魚切割：使用cellpose_run.py開啟GUI界面,把細胞拉進去可以segment
灰階轉換：通過pca.py可以把RGB圖轉成灰階圖
找非合成分裂區域：使用cell_tracking_clean.py先把所有mask分為cell0, cell1, cell2, cell3,再用merge_fi.py把上一步驟mask merge起來,可以用false_posit.py去測試這個merge code效果好壞
從單顆細胞預測此細胞是屬於分裂幾次的預測：使用train_densenet_13.py,train_unet.py,trin_vit.py可以得到3個model

