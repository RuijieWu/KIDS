async fetchActivityData() {
  try {
      const now = new Date();
      const todayStart = new Date(now);
      todayStart.setHours(0, 0, 0, 0); // Start of today

      // Define time intervals for today
      const timeIntervals = [
          "3:00", "6:00", "9:00", "12:00", "15:00", "18:00", "21:00", "24:00"
      ];

      // Format current date for today's intervals
      const formattedDatesToday = [];
      timeIntervals.forEach((interval, index) => {
          const [hours, minutes] = interval.split(':');
          const startTime = new Date(todayStart);
          startTime.setHours(parseInt(hours), parseInt(minutes), 0, 0);
          formattedDatesToday.push(startTime);
      });

      // For yesterday, adjust by subtracting one day
      const yesterdayStart = new Date(todayStart);
      yesterdayStart.setDate(yesterdayStart.getDate() - 1);

      // Format date for yesterday's intervals
      const formattedDatesYesterday = [];
      timeIntervals.forEach((interval, index) => {
          const [hours, minutes] = interval.split(':');
          const startTime = new Date(yesterdayStart);
          startTime.setHours(parseInt(hours), parseInt(minutes), 0, 0);
          formattedDatesYesterday.push(startTime);
      });

      // Fetch data for today's intervals
      const todayDataPromises = formattedDatesToday.map(async (startTime, index) => {
          const endTime = index === formattedDatesToday.length - 1 ? now : formattedDatesToday[index + 1];

          const response = await axios.get('https://ecloud.10086.cn/api/safebox-asp/csproxy/v2/alarmCenter/alarmList', {
              params: {
                  page: 1,
                  pageSize: 1,
                  alarmTypes: [11, 10, 8, 14, 15, 16, 13, 9, 12],
                  attackStartTime: startTime.toISOString(),
                  attackEndTime: endTime.toISOString(),
              },
              headers: {
                  "Content-Type": "application/json",
                  'Accept-Encoding': 'gzip'
                  // Add authorization if needed
              }
          });

          return response.data.total;
      });

      // Fetch data for yesterday's intervals
      const yesterdayDataPromises = formattedDatesYesterday.map(async (startTime, index) => {
          const endTime = index === formattedDatesYesterday.length - 1 ? todayStart : formattedDatesYesterday[index + 1];

          const response = await axios.get('https://ecloud.10086.cn/api/safebox-asp/csproxy/v2/alarmCenter/alarmList', {
              params: {
                  page: 1,
                  pageSize: 1,
                  alarmTypes: [11, 10, 8, 14, 15, 16, 13, 9, 12],
                  attackStartTime: startTime.toISOString(),
                  attackEndTime: endTime.toISOString(),
              },
              headers: {
                  "Content-Type": "application/json",
                  'Accept-Encoding': 'gzip'
                  // Add authorization if needed
              }
          });

          return response.data.total;
      });

      // Wait for all promises to resolve
      const todayCounts = await Promise.all(todayDataPromises);
      const yesterdayCounts = await Promise.all(yesterdayDataPromises);

      // Update activityChart.data with today and yesterday data
      this.activityChart.data.series = [todayCounts, yesterdayCounts];
  } catch (error) {
      console.error('Failed to fetch activity data:', error);
  }
},
