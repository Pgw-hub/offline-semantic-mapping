// Offline replay host for raisin_lidar_slam_plugin.
//
// Loads the unmodified plugin (.so, create()) in this process without a robot node, feeds it the
// recorded /livox/imu and /livox/lidar in recorder-arrival order on the raisin replay clock
// (simulatedClockMode), and dumps everything the plugin publishes for the offline map pipeline.
// Each scan is held until the plugin has answered it on /Odometry/base, so no scan is dropped
// however slow the machine is.
//
// usage: offline_slam_host <in.bin> <out.bin> <plugin.so> <map_name>
//                          <imu tx ty tz qx qy qz qw> <lidar tx ty tz qx qy qz qw>
//   in.bin  : export_sensor_dump.py output (kind, log_time, raisin-serialized body)
//   out.bin : magic "OSLO", records {u8 topic, i64 sim_time_ns, u32 n, bytes} where bytes =
//             8-byte stamp (sim time, like the recorder) + raisin-serialized message
//   tf args : base -> livox_lidar_0:imu / :lidar from the recording's tf_static
// env OSL_SCAN_TIMEOUT_S (default 5): wall seconds to wait for a scan's odometry before moving on
// env OSL_RATE (default 0 = unpaced): cap replay speed, 1 = real time
// env OSL_IMU_LEAD_S (default 0.15): imu published ahead of each scan's header stamp
// HOME must point to a profile whose .raisin/ holds the lidar_slam params (see run.sh).

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "raisin_interfaces/msg/keyframe_cloud.hpp"
#include "raisin_interfaces/srv/save_point_cloud_map.hpp"
#include "raisin_network/network.hpp"
#include "raisin_network/node.hpp"
#include "raisin_network/raisin.hpp"
#include "raisin_plugin/plugin.hpp"
#include "raisin_robot/robot.hpp"
#include "raisin_tf/transform.hpp"
#include "raisin_util/simulated_time.hpp"
#include "raisin_util/time.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_srvs/srv/trigger.hpp"

using namespace raisin;

namespace {

int64_t stampNs(const builtin_interfaces::msg::Time& t) {
  return static_cast<int64_t>(t.sec) * 1000000000LL + t.nanosec;
}

void setSimNs(int64_t ns) {
  builtin_interfaces::msg::Time t;
  splitNanoseconds(ns, t.sec, t.nanosec);
  setSimulatedTime(t);
}

int64_t simNs() { return simulatedTimeNs().load(); }

Eigen::Isometry3d isoFromArgs(char** a) {
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.translation() = Eigen::Vector3d(std::atof(a[0]), std::atof(a[1]), std::atof(a[2]));
  Eigen::Quaterniond q(std::atof(a[6]), std::atof(a[3]), std::atof(a[4]), std::atof(a[5]));
  T.linear() = q.normalized().toRotationMatrix();
  return T;
}

// Output dump: callbacks only serialize; one writer thread does the disk work.
class Dump {
 public:
  explicit Dump(const std::string& path) : out_(path, std::ios::binary) {
    out_.write("OSLO", 4);
    writer_ = std::thread([this] { run(); });
  }
  ~Dump() { close(); }
  template <typename Msg>
  void add(uint8_t topic, const Msg& msg) {
    Rec r;
    r.topic = topic;
    r.t = simNs();
    builtin_interfaces::msg::Time st;
    splitNanoseconds(r.t, st.sec, st.nanosec);
    r.bytes.resize(8);
    std::memcpy(r.bytes.data(), &st.sec, 4);
    std::memcpy(r.bytes.data() + 4, &st.nanosec, 4);
    msg.setBuffer(r.bytes);
    {
      std::lock_guard<std::mutex> l(m_);
      q_.push_back(std::move(r));
      counts_[topic]++;
    }
    cv_.notify_one();
  }
  void close() {
    {
      std::lock_guard<std::mutex> l(m_);
      if (done_) return;
      done_ = true;
    }
    cv_.notify_one();
    if (writer_.joinable()) writer_.join();
    out_.close();
  }
  uint64_t count(uint8_t topic) {
    std::lock_guard<std::mutex> l(m_);
    return counts_[topic];
  }

 private:
  struct Rec {
    uint8_t topic;
    int64_t t;
    std::vector<unsigned char> bytes;
  };
  void run() {
    for (;;) {
      std::unique_lock<std::mutex> l(m_);
      cv_.wait(l, [this] { return done_ || !q_.empty(); });
      if (q_.empty() && done_) return;
      Rec r = std::move(q_.front());
      q_.pop_front();
      l.unlock();
      const uint32_t n = static_cast<uint32_t>(r.bytes.size());
      out_.write(reinterpret_cast<const char*>(&r.topic), 1);
      out_.write(reinterpret_cast<const char*>(&r.t), 8);
      out_.write(reinterpret_cast<const char*>(&n), 4);
      out_.write(reinterpret_cast<const char*>(r.bytes.data()), n);
    }
  }
  std::ofstream out_;
  std::thread writer_;
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Rec> q_;
  uint64_t counts_[8] = {0};
  bool done_ = false;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 19) {
    std::cerr << "usage: " << argv[0]
              << " <in.bin> <out.bin> <plugin.so> <map_name> <imu tx ty tz qx qy qz qw>"
                 " <lidar tx ty tz qx qy qz qw>\n";
    return 1;
  }
  const std::string inPath = argv[1], outPath = argv[2], soPath = argv[3], mapName = argv[4];
  const Eigen::Isometry3d T_b_imu = isoFromArgs(argv + 5), T_b_lidar = isoFromArgs(argv + 12);
  const double scanTimeoutS = std::getenv("OSL_SCAN_TIMEOUT_S") ? std::atof(std::getenv("OSL_SCAN_TIMEOUT_S")) : 5.0;

  std::ifstream in(inPath, std::ios::binary);
  char magic[4];
  uint32_t ver = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(&ver), 4);
  if (!in || std::memcmp(magic, "OSLD", 4) != 0 || ver != 1) {
    std::cerr << "bad input " << inPath << "\n";
    return 1;
  }
  auto readRec = [&](uint8_t& kind, int64_t& t, std::vector<unsigned char>& b) {
    uint32_t n = 0;
    if (!in.read(reinterpret_cast<char*>(&kind), 1)) return false;
    in.read(reinterpret_cast<char*>(&t), 8);
    in.read(reinterpret_cast<char*>(&n), 4);
    b.resize(n);
    in.read(reinterpret_cast<char*>(b.data()), n);
    return static_cast<bool>(in);
  };

  // Replay clock first: timers created after this count periods on the recording's time.
  raisinInit();
  simulatedClockMode().store(true);
  {
    uint8_t k;
    int64_t t;
    std::vector<unsigned char> b;
    auto pos = in.tellg();
    readRec(k, t, b);
    setSimNs(t - 1000000000LL);  // one second before the first message
    in.seekg(pos);
  }

  // The robot node's thread layout (config_snapshot effective_params threads): lidar and imu on
  // separate workers -- a scan waits up to 50 ms on the imu thread for imu past its last point
  // (findImuRange), which cannot arrive if both share one worker.
  std::vector<std::vector<std::string>> threads = {
    {"main"}, {"lidar"}, {"imu"}, {"pgo"}, {"submap"}, {"plugin"}, {"capture"}};
  auto network = std::make_shared<Network>("offline_slam", "offline", threads);
  auto& gr = GlobalResource::getInstance();
  gr.network = network;
  gr.transform = std::make_shared<tf::Transform>(network);
  const std::string id = network->getMyId();

  raisim::World worldHub, worldSim;
  raisim::RaisimServer serverHub(&worldHub), serverSim(&worldSim);

  void* so = dlopen(soPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!so) {
    std::cerr << "dlopen: " << dlerror() << "\n";
    return 1;
  }
  using CreateFn = plugin::Plugin* (*)(raisim::World&, raisim::RaisimServer&, raisim::World&,
    raisim::RaisimServer&, GlobalResource&);
  auto create = reinterpret_cast<CreateFn>(dlsym(so, "create"));
  if (!create) {
    std::cerr << "dlsym create: " << dlerror() << "\n";
    return 1;
  }
  plugin::Plugin* slam = create(worldHub, serverHub, worldSim, serverSim, gr);
  if (!slam || !slam->init()) {
    std::cerr << "plugin init failed\n";
    return 1;
  }
  // After init(): it publishes an identity <id> -> <id>/livox_lidar_0:imu placeholder for the GUI,
  // which would replace ours. start_mapping then reads base -> imu (loadExtrinsicsFromTf).
  gr.transform->sendTransform(id, id + "/base", Eigen::Isometry3d::Identity());
  gr.transform->sendTransform(id + "/base", id + "/livox_lidar_0:imu", T_b_imu, true);
  gr.transform->sendTransform(id + "/base", id + "/livox_lidar_0:lidar", T_b_lidar, true);
  if (!gr.transform->canTransform(id + "/base", id + "/livox_lidar_0:imu")) {
    std::cerr << "TF base -> livox_lidar_0:imu not resolvable\n";
    return 1;
  }

  Node host(network);
  auto pubImu = host.createPublisher<sensor_msgs::msg::Imu>("livox/imu", 100000);
  auto pubLidar = host.createPublisher<sensor_msgs::msg::PointCloud2>("livox/lidar", 100);

  Dump dump(outPath);
  std::mutex odomM;
  std::condition_variable odomCv;
  int64_t lastOdomStamp = 0;
  auto subOdom = host.createSubscriber<nav_msgs::msg::Odometry>("/Odometry/base", nullptr,
    [&](nav_msgs::msg::Odometry::ConstSharedPtr m) {
      dump.add(0, *m);
      {
        std::lock_guard<std::mutex> l(odomM);
        lastOdomStamp = stampNs(m->header.stamp);
      }
      odomCv.notify_all();
    }, "capture");
  auto subCloud = host.createSubscriber<sensor_msgs::msg::PointCloud2>("/cloud_registered", nullptr,
    [&](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { dump.add(1, *m); }, "capture");
  auto subPath = host.createSubscriber<nav_msgs::msg::Path>("/path/pgo", nullptr,
    [&](nav_msgs::msg::Path::ConstSharedPtr m) { dump.add(2, *m); }, "capture");
  auto subKf = host.createSubscriber<raisin_interfaces::msg::KeyframeCloud>("/pgo/keyframe_cloud",
    nullptr, [&](raisin_interfaces::msg::KeyframeCloud::ConstSharedPtr m) { dump.add(3, *m); },
    "capture");
  auto subLio = host.createSubscriber<nav_msgs::msg::Odometry>("/Odometry/lio", nullptr,
    [&](nav_msgs::msg::Odometry::ConstSharedPtr m) { dump.add(4, *m); }, "capture");

  auto callTrigger = [&](const std::string& srv) {
    auto c = host.createClient<std_srvs::srv::Trigger>(srv, nullptr);
    if (!c->waitForService(std::chrono::seconds(10))) {
      std::cerr << "no service " << srv << "\n";
      return false;
    }
    auto rq = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto f = c->asyncSendRequest(rq);
    if (f.wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
      std::cerr << srv << " timed out\n";
      return false;
    }
    std::cerr << srv << ": " << f.get()->message << "\n";
    return f.get()->success;
  };
  std::this_thread::sleep_for(std::chrono::milliseconds(500));  // let local pub/sub connect
  if (!callTrigger("start_mapping")) return 1;

  // Feed in recorder-arrival order, except that before a scan goes out every imu up to
  // OSL_IMU_LEAD_S (default 0.15 s) past its header stamp is published first: on the robot the scan
  // arrives ~100 ms after its first point and the deskew waits for imu past its last point.
  const double rate = std::getenv("OSL_RATE") ? std::atof(std::getenv("OSL_RATE")) : 0.0;  // 0 = as fast as the plugin answers
  const double imuLeadS = std::getenv("OSL_IMU_LEAD_S") ? std::atof(std::getenv("OSL_IMU_LEAD_S")) : 0.15;
  struct Pending {
    int64_t t;
    sensor_msgs::msg::Imu m;
  };
  std::deque<Pending> imuAhead;   // read ahead of the current scan, not yet published
  std::deque<std::pair<int64_t, std::vector<unsigned char>>> scansAhead;
  bool eof = false;
  auto readMore = [&]() {
    uint8_t k;
    int64_t tt;
    std::vector<unsigned char> b;
    if (!readRec(k, tt, b)) {
      eof = true;
      return;
    }
    if (k == 0) {
      Pending p{tt, {}};
      p.m.getBuffer(b.data());
      imuAhead.push_back(std::move(p));
    } else {
      scansAhead.emplace_back(tt, std::move(b));
    }
  };
  uint64_t nImu = 0, nScan = 0, nAnswered = 0, nTimeout = 0;
  const auto wall0 = std::chrono::steady_clock::now();
  int64_t tFirst = 0, tLast = 0, t = 0;
  auto publishImu = [&](Pending& p) {
    if (p.t > simNs()) setSimNs(p.t);
    pubImu->publish(p.m);
    ++nImu;
  };
  for (;;) {
    while (!eof && scansAhead.empty()) readMore();
    if (scansAhead.empty()) break;
    auto [tScan, buf] = std::move(scansAhead.front());
    scansAhead.pop_front();
    sensor_msgs::msg::PointCloud2 m;
    m.getBuffer(buf.data());
    const int64_t scanStamp = stampNs(m.header.stamp);
    const int64_t needImu = scanStamp + static_cast<int64_t>(imuLeadS * 1e9);
    // imu that arrived before this scan, then imu up to the lead time
    for (;;) {
      while (!eof && imuAhead.empty()) readMore();
      if (imuAhead.empty()) break;
      Pending& p = imuAhead.front();
      if (p.t > tScan && stampNs(p.m.header.stamp) > needImu) break;
      publishImu(p);
      imuAhead.pop_front();
    }
    t = tScan;
    if (!tFirst) tFirst = t;
    tLast = t;
    if (t > simNs()) setSimNs(t);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));  // imu worker drains first
    pubLidar->publish(m);
    ++nScan;
    std::unique_lock<std::mutex> l(odomM);
    // Before the first odometry (imu calibration, first scan) the plugin answers nothing; don't stall.
    const double waitS = lastOdomStamp == 0 ? 0.3 : scanTimeoutS;
    if (odomCv.wait_for(l, std::chrono::duration<double>(waitS),
          [&] { return lastOdomStamp >= scanStamp; })) {
      ++nAnswered;
    } else {
      ++nTimeout;
    }
    if (rate > 0) {  // OSL_RATE: cap replay speed (1 = real time)
      const double ahead = (t - tFirst) / 1e9 / rate - std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
      if (ahead > 0) std::this_thread::sleep_for(std::chrono::duration<double>(ahead));
    }
    if (nScan % 500 == 0) {
      const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
      std::cerr << "scan " << nScan << " answered " << nAnswered << " timeout " << nTimeout
                << " data " << (t - tFirst) / 1e9 << " s wall " << wall << " s\n";
    }
  }
  // Two more seconds of data time so the 1 Hz loop closure and 0.5 Hz path publisher run once more.
  for (int i = 0; i < 40; ++i) {
    setSimNs(tLast + (i + 1) * 50000000LL);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  {
    auto c = host.createClient<raisin_interfaces::srv::SavePointCloudMap>("save_point_cloud_map", nullptr);
    if (c->waitForService(std::chrono::seconds(10))) {
      auto rq = std::make_shared<raisin_interfaces::srv::SavePointCloudMap::Request>();
      rq->name = mapName;
      auto f = c->asyncSendRequest(rq);
      if (f.wait_for(std::chrono::seconds(300)) == std::future_status::ready)
        std::cerr << "save_point_cloud_map: " << f.get()->success << " " << f.get()->message << "\n";
      else
        std::cerr << "save_point_cloud_map timed out\n";
    }
  }
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  std::cerr << "RESULT imu " << nImu << " scans " << nScan << " answered " << nAnswered << " timeout "
            << nTimeout << " | out odom " << dump.count(0) << " cloud " << dump.count(1) << " path "
            << dump.count(2) << " keyframe " << dump.count(3) << " | data " << (tLast - tFirst) / 1e9
            << " s wall " << wall << " s\n";
  dump.close();
  std::_Exit(0);  // skip plugin/network teardown order issues; everything is on disk
}
