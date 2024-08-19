async fetchChartData() {
    try {
        const today = new Date();
        const formattedDates_kairos = ["2018-04-01 00:00:00","2018-04-02 00:00:00","2018-04-03 00:00:00","2018-04-04 00:00:00",
            "2018-04-05 00:00:00","2018-04-06 00:00:00","2018-04-07 00:00:00","2018-04-08 00:00:00","2018-04-09 00:00:00","2018-04-10 00:00:00",
            "2018-04-11 00:00:00","2018-04-12 00:00:00"
        ];
        const formattedDates_kairos_day = ["04-01","04-02","04-03","04-04",
            "04-05","04-06","04-07","04-08","04-09","04-10",
            "04-11","04-12"
        ];
        today.setHours(0, 0, 0, 0); // 将时间设为当天的开始

        for (let i = 0; i < 7; i++) {
            const day = new Date(today);
            day.setDate(today.getDate() - i);
            const formattedDate = (`0${day.getMonth() + 1}`).slice(-2) + '-' + (`0${day.getDate()}`).slice(-2); // 格式化日期为 MM-DD
            this.usersChart.data.labels[6 - i] = formattedDates_kairos_day[11 - i];
        }

        const formattedDates = []; // 用于存储格式化后的日期时间
        for (let i = 0; i < this.usersChart.data.labels.length; i++) {
            const mmDd = this.usersChart.data.labels[i]; // 取出 MM-DD 格式的日期
            const [month, day] = mmDd.split('-'); // 分割月份和日期
            const now = new Date();
            const year = now.getFullYear();
            const formattedDate = `${year}-${(`0${month}`).slice(-2)}-${(`0${day}`).slice(-2)} 00:00:00`;
            formattedDates.push(formattedDate);
        }

        const now = new Date();
        const year = now.getFullYear();
        const currentMonth = (`0${now.getMonth() + 1}`).slice(-2); // 月份从 0 开始，所以加 1
        const currentDay = (`0${now.getDate()}`).slice(-2);
        const hours = (`0${now.getHours()}`).slice(-2);
        const minutes = (`0${now.getMinutes()}`).slice(-2);
        const seconds = (`0${now.getSeconds()}`).slice(-2);
        const currentFormattedDate = `${year}-${currentMonth}-${currentDay} ${hours}:${minutes}:${seconds}`;
        formattedDates.push(currentFormattedDate);

        const kairosPromises = [];
        const openPromises = [];

        for (let j = 1; j <= 3; j++) {
            for (let i = 0; i < 7; i++) {
                const k = i * j;
                const attackEndTime = formattedDates[7 - i];
                const attackStartTime = formattedDates[6 - i];
                openPromises.push(
                    axios.get('http://43.138.200.89:8080/alarm/alarm/list', {
                        params: {
                            page: '1',
                            pageSize: '1',
                            alarmTypes: '11,10,8,14,15,16,13,9,12',
                            attackStartTime: attackStartTime,
                            attackEndTime: attackEndTime,
                            securitys: j.toString(),
                        },
                        headers: {
                            "content-type": "application/json",
                        }
                    })
                );
                
                const attackStartTime_kairos = formattedDates_kairos[11 - i];
                const attackEndTime_kairos = formattedDates_kairos[10 - i];
                kairosPromises.push(
                    axios.get('http://43.138.200.89:8080/alarm/alarm/list', {
                        params: {
                            page: '1',
                            pageSize: '1',
                            start_time: attackStartTime_kairos,
                            end_time: attackEndTime_kairos,
                        },
                        headers: {
                            "content-type": "application/json",
                        }
                    })
                );
            }
        }

        const openResponses = await Promise.all(openPromises);
        const kairosResponses = await Promise.all(kairosPromises);

        let kairosIndex = 0;
        let openIndex = 0;

        for (let j = 1; j <= 3; j++) {
            for (let i = 0; i < 7; i++) {
                const response_open = openResponses[openIndex++];
                const response_kairos = kairosResponses[kairosIndex++];
                
                const total_kairos = response_kairos.data.anomalous_actions.total + response_kairos.data.dangerous_actions.total;
                const alarms = response_open.data.body.content;
                const total_open = response_open.data.body.total;

                this.usersChart.data.series[j - 1][6 - i] = total_open + total_kairos;
                this.statsCards[2].value += total_open + total_kairos;
            }
        }

        const totalAlterToday = this.usersChart.data.series[0][6] + this.usersChart.data.series[1][6] + this.usersChart.data.series[2][6];
        this.preferencesChart.data.series[0] = Math.floor(this.usersChart.data.series[0][6] * 100 / totalAlterToday);
        this.preferencesChart.data.series[1] = Math.floor(this.usersChart.data.series[1][6] * 100 / totalAlterToday);
        this.preferencesChart.data.series[2] = 100 - this.preferencesChart.data.series[0] - this.preferencesChart.data.series[1];
        this.preferencesChart.data.labels[0] = this.preferencesChart.data.series[0].toString() + '%';
        this.preferencesChart.data.labels[1] = this.preferencesChart.data.series[1].toString() + '%';
        this.preferencesChart.data.labels[2] = this.preferencesChart.data.series[2].toString() + '%';
    } catch (error) {
        this.error = '获取图表数据失败';
    } finally {
    }
}
