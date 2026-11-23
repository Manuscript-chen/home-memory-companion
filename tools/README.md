# 工具

本目录不放会改记忆的程序。节点 B 到 H 的代码在各自的目录里。

## go2rtc（备用，默认不装）

计划书节点 B：摄像头限制同时连接数、出现「连接数已满」时，才用 go2rtc 把两路 RTSP 转推给本程序。它不替换状态机，也不写入 `events`。

原先仓库里提交过 Windows 安装包 `go2rtc_win64.zip`（两份，分别在 `downloads/` 和 `工具/downloads/`）。压缩包已从当前版本删除，请不要再提交。

需要时到 <https://github.com/AlexxIT/go2rtc/releases> 下载对应系统的压缩包，解压到本机任意目录运行。`.gitignore` 会忽略 `*.zip`。

Frigate 只适合安装时看画面，它自己的物体事件不是本项目的状态机结果，不要写进数据库。
