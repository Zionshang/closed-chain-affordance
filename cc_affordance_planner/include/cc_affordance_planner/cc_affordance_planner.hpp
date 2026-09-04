///////////////////////////////////////////////////////////////////////////////
//      Title     : cc_affordance_planner.hpp
//      Project   : cc_affordance_planner
//      Created   : Fall 2023
//      Author    : Janak Panthi (Crasun Jans)
///////////////////////////////////////////////////////////////////////////////

#ifndef CC_AFFORDANCE_PLANNER
#define CC_AFFORDANCE_PLANNER

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <affordance_util/affordance_util.hpp>
#include <cc_affordance_planner/rm_cca.hpp>
#include <chrono>
#include <optional>
#include <vector>
#include <thread>

namespace cc_affordance_planner
{

/**
* @brief Enum describing common ee orientation constraints
*/
enum class EeOrientationConstraint 
{
    PRESERVE,
    DEFAULT
};

/**
 * @brief Enum describing various planning types that the closed-chain affordance planner offers
 */
enum class PlanningType
{
    APPROACH,
    AFFORDANCE,
    EE_ORIENTATION_ONLY,
    CARTESIAN_GOAL
};

/**
 * @brief Enum describing various motion types that the closed-chain affordance model offers
 */
enum class MotionType
{
    APPROACH,
    AFFORDANCE,
};

/**
 * @brief Struct describing the goals for the Closed-chain affordance planner in terms of affordance state, ee
 * orientation state, canonical_pose, and gripper state.
 */
struct Goal
{

    double affordance = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd ee_orientation;
    Eigen::Matrix4d canonical_pose = Eigen::Matrix4d::Constant(std::numeric_limits<double>::quiet_NaN());
    double gripper = std::numeric_limits<double>::quiet_NaN();
};

/**
 * @brief Struct describing a task for the Closed-Chain Affordance planner in terms of affordance info, goal state,
 * trajectory density, motion type, virtual screw order, canonical pose, and gripper goal type.
 */
struct TaskDescription
{

    affordance_util::ScrewInfo affordance_info;
    Goal goal;
    int trajectory_density = 10;
    double approach_gamma = 1.0; // m/rad, weighs rotational motion relative to translational motion along the approach trajectory
    MotionType motion_type = MotionType::AFFORDANCE;
    affordance_util::VirtualScrewOrder vir_screw_order = affordance_util::VirtualScrewOrder::XYZ;
    affordance_util::GripperGoalType gripper_goal_type = affordance_util::GripperGoalType::CONSTANT;
    EeOrientationConstraint ee_orientation_constraint = EeOrientationConstraint::DEFAULT;
    affordance_util::ScrewInfoFrom affordance_info_from; // Get affordance info from specified method
    affordance_util::PoseFrom canonical_pose_from; // Get canonical pose from specified method
    affordance_util::ReserveMobilityDescription reserve_mobility; // Disabled by default; enables RM-CCA when true

    /**
     * @brief Given a planning type, constructs a cca task description with necessary parameters. This constructor is
     * especially useful for CARTESIAN_GOAL and EE_ORIENTATION_ONLY planning, which are special cases of APPROACH and
     * AFFORDANCE motions respectively.
     *
     * @param task_type cc_affordance_planner::PlanningType indicating what type of planning is intended.
     */
    explicit TaskDescription(const PlanningType &planningType);

    /**
     * @brief Default constructor for cca task description
     */
    TaskDescription() = default;
};

/**
 * @brief Enum qualitatively describing a trajectory length as FULL, PARTIAL, or UNSET
 */
enum class TrajectoryDescription
{
    FULL,
    PARTIAL,
    UNSET
};

/**
 * @brief Enum representing update methods for the closed-chain affordance planner
 */
enum class UpdateMethod
{
    INVERSE,
    TRANSPOSE,
    BEST
};

/**
 * @brief Designed to contain the result of the Closed-chain Affordance planner with fields:
 * success indicating success; trajectory_description indicating full or partial trajectory with values, "FULL",
 * "PARTIAL", or "UNSET"; joint_trajectory representing the joint trajectory; planning_time representing time taken for
 * planning in microseconds; update_method indicating the type of update scheme used, for instance "inverse",
 * "inverse with dls", and "transpose"; includes_gripper_trajectory indicating whether the result includes gripper
 * trajectory; and task_description representing the task description used for planning.
 */
struct PlannerResult
{
    bool success = false;
    TrajectoryDescription trajectory_description = TrajectoryDescription::UNSET;
    std::vector<Eigen::VectorXd> joint_trajectory;
    std::chrono::microseconds planning_time{0};
    UpdateMethod update_method = UpdateMethod::INVERSE;
    std::string update_trail = "";
    bool includes_gripper_trajectory = false;
    TaskDescription task_description;
    bool reserve_mobility_used = false;                       ///< True if the hierarchy retained reserve motion.
    size_t reserve_activation_count = 0;                      ///< Number of inactive-to-active trajectory transitions.
    std::vector<Eigen::VectorXd> reserve_trajectory;          ///< Reserve generalized state, separate from arm joints.
    std::vector<Eigen::Matrix4d> reserve_pose_trajectory;     ///< Product-of-exponentials reserve pose trajectory.
    std::vector<double> residual_mobility_norm;               ///< Maximum reserve correction norm per point solve.
    std::vector<bool> reserve_active;                         ///< Whether reserve was used at each trajectory point.
    std::vector<size_t> active_arm_dof_count;                 ///< Minimum feasible non-reserve DOF per point solve.
};

/**
 * @brief Designed to contain the configuration settings for the Closed-chain Affordance planner with fields:
 * accuracy indicating the accuracy of the planner as percentage; closure_err_threshold indicating the threshold for the
 * closed-chain closure error; and ik_max_itr indicating the maximum interations for the closed-chain IK solver.
 */
struct PlannerConfig
{
    double accuracy = 10.0 / 100.0;
    double closure_err_threshold_ang = 1e-4;
    double closure_err_threshold_lin = 1e-5;
    int ik_max_itr = 200;
    UpdateMethod update_method = UpdateMethod::BEST;
    bool enable_joint_limits = true;          ///< Enforce RM-CCA primary-coordinate bounds with an active set.
    bool enable_nullspace_planning = true;    ///< Prefer stationary reserve mobility in the CCA task null space.
    double svd_relative_tolerance = 1e-8;       ///< Relative singular-value cutoff used by RM-CCA.
    double residual_mobility_tolerance = 1e-10; ///< Threshold for classifying a reserve correction as active.
    double joint_limit_margin = 1e-6;           ///< Inward margin applied to absolute arm joint limits.
    double joint_limit_tolerance = 1e-10;       ///< Numerical tolerance for active-set boundary detection.
};

/**
 * @brief Base Class for the Closed-Chain Affordance Planner
 */
class CcAffordancePlanner
{
  public:
    // Constructor
    explicit CcAffordancePlanner(const PlannerConfig &plannerConfig = PlannerConfig());

    /**
     * @brief Enables strict CCA-first/base-stationarity Newton updates with an explicit primary partition.
     *
     * @param reserve_mobility Generic reserve description used for limits, step limiting, and pose diagnostics.
     * @param arm_start_absolute Absolute arm state corresponding to zero arm differential state.
     * @param arm_lower_limits Absolute arm lower limits.
     * @param arm_upper_limits Absolute arm upper limits.
     * @param reserve_primary_indices Reserve columns in the primary part of the closed-chain model.
     * @param arm_primary_indices Arm columns in the primary part of the closed-chain model.
     */
    void configure_reserve_mobility(const affordance_util::ReserveMobilityDescription &reserve_mobility,
                                    const Eigen::VectorXd &arm_start_absolute,
                                    const Eigen::VectorXd &arm_lower_limits,
                                    const Eigen::VectorXd &arm_upper_limits,
                                    const std::vector<size_t> &reserve_primary_indices,
                                    const std::vector<size_t> &arm_primary_indices);

    /** @brief Disables RM-CCA and restores the legacy fixed-base solver path. */
    void disable_reserve_mobility();

    // Methods
    /**
     * @brief After setting the planner parameters described in the class documentation, call this function to generate
     * the differential joint trajectory to execute a given approach motion.
     *
     * @param slist Eigen::MatrixXd containing as columns 6x1 Screws representing all joints of the closed-chain model,
     * i.e. robot joints, virtual ee joint, affordance joint
     * @param theta_sdf Eigen::VectorXd containing secondary joint angle goals including EE orientation and affordance
     * such that the affordance goal is the end element.
     * @param task_offset_tau A numeric parameter indicating the length of the secondary joint vector.
     * The value 1 implies only affordance control, 2 represents affordance control
     * along with fixing the gripper x-axis, 3 involves fixing the gripper x and y axes,
     * and 4 involves fixing the gripper x, y, and z axes.
     * @param stepper_max_itr_m int denoting the density for the trajectory in terms of number of points
     *
     * @return Struct containing the result of the planning with fields: success indicating success;
     * traj_full_or_partial indicating full or partial trajectory with values, "Full", "Partial", or "Unset";
     * joint_trajectory representing the joint trajectory; and planning_time representing time taken for planning in
     * microseconds.
     */
    PlannerResult generate_approach_motion_joint_trajectory(const Eigen::MatrixXd &slist,
                                                            const Eigen::VectorXd &theta_sdf,
                                                            const size_t &task_offset_tau,
                                                            const int &stepper_max_itr_m);

    /**
     * @brief After setting the planner parameters described in the class documentation, call this function to generate
     * the differential joint trajectory to execute a given approach motion.
     *
     * @param slist Eigen::MatrixXd containing as columns 6x1 Screws representing all joints of the closed-chain model,
     * i.e. robot joints, virtual ee joint, affordance joint
     * @param theta_sdf Eigen::VectorXd containing secondary joint angle goals including EE orientation and affordance
     * such that the affordance goal is the end element.
     * @param task_offset_tau A numeric parameter indicating the length of the secondary joint vector.
     * The value 1 implies only affordance control, 2 represents affordance control
     * along with fixing the gripper x-axis, 3 involves fixing the gripper x and y axes,
     * and 4 involves fixing the gripper x, y, and z axes.
     * @param stepper_max_itr_m int denoting the density for the trajectory in terms of number of points
     * @param st std::stop_token for cooperative interruption when multi-threading
     *
     * @return Struct containing the result of the planning with fields: success indicating success;
     * traj_full_or_partial indicating full or partial trajectory with values, "Full", "Partial", or "Unset";
     * joint_trajectory representing the joint trajectory; and planning_time representing time taken for planning in
     * microseconds.
     */
    PlannerResult generate_approach_motion_joint_trajectory(const Eigen::MatrixXd &slist,
                                                            const Eigen::VectorXd &theta_sdf,
                                                            const size_t &task_offset_tau, const int &stepper_max_itr_m,
                                                            std::stop_token st);
    /**
     * @brief After setting the planner parameters described in the class documentation, call this function to generate
     * the differential joint trajectory to execute a given affordance.
     *
     * @param slist Eigen::MatrixXd containing as columns 6x1 Screws representing all joints of the closed-chain model,
     * i.e. robot joints, virtual ee joint, affordance joint
     * @param theta_sdf Eigen::VectorXd containing secondary joint angle goals including EE orientation and affordance
     * such that the affordance goal is the end element.
     * @param task_offset_tau A numeric parameter indicating the length of the secondary joint vector.
     * The value 1 implies only affordance control, 2 represents affordance control
     * along with fixing the gripper x-axis, 3 involves fixing the gripper x and y axes,
     * and 4 involves fixing the gripper x, y, and z axes.
     * @param stepper_max_itr_m int denoting the density for the trajectory in terms of number of points
     *
     * @return Struct containing the result of the planning with fields: success indicating success;
     * traj_full_or_partial indicating full or partial trajectory with values, "Full", "Partial", or "Unset";
     * joint_trajectory representing the joint trajectory; and planning_time representing time taken for planning in
     * microseconds.
     */
    PlannerResult generate_affordance_motion_joint_trajectory(const Eigen::MatrixXd &slist,
                                                              const Eigen::VectorXd &theta_sdf,
                                                              const size_t &task_offset_tau,
                                                              const int &stepper_max_itr_m);

    /**
     * @brief After setting the planner parameters described in the class documentation, call this function to generate
     * the differential joint trajectory to execute a given affordance.
     *
     * @param slist Eigen::MatrixXd containing as columns 6x1 Screws representing all joints of the closed-chain model,
     * i.e. robot joints, virtual ee joint, affordance joint
     * @param theta_sdf Eigen::VectorXd containing secondary joint angle goals including EE orientation and affordance
     * such that the affordance goal is the end element.
     * @param task_offset_tau A numeric parameter indicating the length of the secondary joint vector.
     * The value 1 implies only affordance control, 2 represents affordance control
     * along with fixing the gripper x-axis, 3 involves fixing the gripper x and y axes,
     * and 4 involves fixing the gripper x, y, and z axes.
     * @param st std::stop_token for cooperative interruption when multi-threading
     *
     * @return Struct containing the result of the planning with fields: success indicating success;
     * traj_full_or_partial indicating full or partial trajectory with values, "Full", "Partial", or "Unset";
     * joint_trajectory representing the joint trajectory; and planning_time representing time taken for planning in
     * microseconds.
     */
    PlannerResult generate_affordance_motion_joint_trajectory(const Eigen::MatrixXd &slist,
                                                              const Eigen::VectorXd &theta_sdf,
                                                              const size_t &task_offset_tau,
                                                              const int &stepper_max_itr_m, std::stop_token st);

    /**
     * @brief Given a list of closed-chain Screws, initial primary and secondary joint guesses, and desired secondary
     * joint goals, computes the closed-chain inverse kinematics solution.
     *
     * @param slist Eigen::MatrixXd containing as columns 6x1 Screws of the closed-chain mechanism
     * @param theta_pg Eigen::VectorXd containing guesses for the primary joints
     * @param theta_sg Eigen::VectorXd containing guesses for the secondary joints
     * @param theta_sd Eigen::VectorXd containing goals for the secondary joints
     *
     * @return Eigen::VectorXd containing the inverse kinematics solution (including the exact secondary joint goals as
     * well)
     */
    std::optional<Eigen::VectorXd> call_cc_ik_solver(const Eigen::MatrixXd &slist, const Eigen::VectorXd &theta_pg,
                                                     const Eigen::VectorXd &theta_sg, const Eigen::VectorXd &theta_sd);

    /**
     * @brief Given a list of closed-chain Screws, initial primary and secondary joint guesses, and desired secondary
     * joint goals, computes the closed-chain inverse kinematics solution.
     *
     * @param slist Eigen::MatrixXd containing as columns 6x1 Screws of the closed-chain mechanism
     * @param theta_pg Eigen::VectorXd containing guesses for the primary joints
     * @param theta_sg Eigen::VectorXd containing guesses for the secondary joints
     * @param theta_sd Eigen::VectorXd containing goals for the secondary joints
     * @param st std::stop_token for cooperative interruption when multi-threading
     *
     * @return Eigen::VectorXd containing the inverse kinematics solution (including the exact secondary joint goals as
     * well)
     */
    std::optional<Eigen::VectorXd> call_cc_ik_solver(const Eigen::MatrixXd &slist, const Eigen::VectorXd &theta_pg,
                                                     const Eigen::VectorXd &theta_sg, const Eigen::VectorXd &theta_sd,
                                                     std::stop_token st);

  protected:
    size_t nof_pjoints_;            // number of primary joints
    size_t nof_sjoints_;            // number of secondary joints
    double cond_N_threshold_ = 100; // condition number threshold for N to be considered singular
    bool dls_flag_ = false;         // flag indicating whether DLS method was used
    constexpr static double lambda_ = 1.1; // damping factor for the DLS method
    constexpr static double goal_min_ = 1e-5; // minimum magnitude to clamp secondary goals to avoid numerical issues

  private:
    struct RmIkDiagnostics
    {
        bool reserve_active = false;
        double residual_norm = 0.0;
        size_t active_arm_dof_count = 0;
    };

    //--Planner config parameters
    double accuracy_; // accuracy of the secondary goals
    double eps_rw_;   // closure error threshold for angular part
    double eps_rv_;   // closure error threshold for linear part
    int max_itr_l_;   // max interations for IK solver
    bool enable_joint_limits_;
    bool enable_nullspace_planning_;
    double svd_relative_tolerance_;
    double residual_mobility_tolerance_;
    double joint_limit_margin_;
    double joint_limit_tolerance_;
    //--EOF Planner config parameters
    constexpr static size_t twist_length_ = 6; // length of a twist vector
    Eigen::VectorXd theta_s_tol_;              // IK tolerance for secondary joint goal vector
    bool reserve_mobility_enabled_ = false;
    affordance_util::ReserveMobilityDescription reserve_mobility_;
    Eigen::VectorXd arm_start_absolute_;
    Eigen::VectorXd arm_lower_limits_;
    Eigen::VectorXd arm_upper_limits_;
    std::vector<size_t> reserve_primary_indices_;
    std::vector<size_t> arm_primary_indices_;
    RmIkDiagnostics last_rm_ik_diagnostics_;

    std::optional<Eigen::VectorXd> call_rm_cc_ik_solver(const Eigen::MatrixXd &slist,
                                                        const Eigen::VectorXd &theta_pg,
                                                        const Eigen::VectorXd &theta_sg,
                                                        const Eigen::VectorXd &theta_sd,
                                                        std::stop_token st);
    Eigen::VectorXd make_primary_start_guess() const;
    void append_trajectory_point(PlannerResult &result, const Eigen::VectorXd &closed_chain_point) const;

    /**
     * @brief Given a list of closed-chain screw axes, primary and secondary network matrices, and primary and secondary
     * joint angles, returns by reference joint angles corrected for closed-chain closure error.
     *
     * @param slist Eigen::MatrixXd containing as columns 6x1 Screws of the closed-chain mechanism
     * @param Np Eigen::MatrixXd representing primary network matrix
     * @param Ns Eigen::MatrixXd representing secondary network matrix
     * @param theta_p Eigen::VectorXd containing primary joint angles
     * @param theta_s Eigen::VectorXd containing secondary joint angles
     */
    void adjust_for_closure_error(const Eigen::MatrixXd &slist, const Eigen::MatrixXd &Np, const Eigen::MatrixXd &Ns,
                                  Eigen::VectorXd &theta_p, Eigen::VectorXd &theta_s, Eigen::VectorXd &rho);

    /**
     * @brief Given the primary joint angles, desired and current secondary joint angles, and constraint Jacobian,
     * returns by reference updated primary joint angles. This function must be overriden in child class implementation.
     *
     * @param theta_p Eigen::VectorXd containing primary joint angles
     * @param theta_sd Eigen::VectorXd containing desired secondary joint angles
     * @param theta_s Eigen::VectorXd containing secondary joint angles
     * @param N Eigen::MatrixXd representing the constraint Jacobian
     */
    virtual void update_theta_p(Eigen::VectorXd &theta_p, const Eigen::VectorXd &theta_sd,
                                const Eigen::VectorXd &theta_s, const Eigen::MatrixXd &N) = 0;
};

/**
 * @brief Child class for the Closed-Chain Affordance Planner implementing the transpose method
 */
class CcAffordancePlannerTranspose : public CcAffordancePlanner
{
  public:
    explicit CcAffordancePlannerTranspose() : CcAffordancePlanner() {}
    explicit CcAffordancePlannerTranspose(const PlannerConfig &plannerConfig) : CcAffordancePlanner(plannerConfig) {}

  private:
    /**
     * @brief Given the primary joint angles, desired and current secondary joint angles, and constraint Jacobian,
     * returns by reference updated primary joint angles using the transpose method.
     *
     * @param theta_p Eigen::VectorXd containing primary joint angles
     * @param theta_sd Eigen::VectorXd containing desired secondary joint angles
     * @param theta_s Eigen::VectorXd containing secondary joint angles
     * @param N Eigen::MatrixXd representing the constraint Jacobian
     */
    virtual void update_theta_p(Eigen::VectorXd &theta_p, const Eigen::VectorXd &theta_sd,
                                const Eigen::VectorXd &theta_s, const Eigen::MatrixXd &N) override;
};

/**
 * @brief Child class for the Closed-Chain Affordance Planner implementing the inverse method
 */
class CcAffordancePlannerInverse : public CcAffordancePlanner
{
  public:
    explicit CcAffordancePlannerInverse() : CcAffordancePlanner() {}
    explicit CcAffordancePlannerInverse(const PlannerConfig &plannerConfig) : CcAffordancePlanner(plannerConfig) {}

  private:
    /**
     * @brief Given the primary joint angles, desired and current secondary joint angles, and constraint Jacobian,
     * returns by reference updated primary joint angles using the pseudoinverse method or DLS if near singularities.
     *
     * @param theta_p Eigen::VectorXd containing primary joint angles
     * @param theta_sd Eigen::VectorXd containing desired secondary joint angles
     * @param theta_s Eigen::VectorXd containing secondary joint angles
     * @param N Eigen::MatrixXd representing the constraint Jacobian
     */
    virtual void update_theta_p(Eigen::VectorXd &theta_p, const Eigen::VectorXd &theta_sd,
                                const Eigen::VectorXd &theta_s, const Eigen::MatrixXd &N) override;
};

} // namespace cc_affordance_planner
#endif // CC_AFFORDANCE_PLANNER
