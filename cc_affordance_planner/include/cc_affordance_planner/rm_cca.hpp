///////////////////////////////////////////////////////////////////////////////
//      Title     : rm_cca.hpp
//      Project   : cc_affordance_planner
///////////////////////////////////////////////////////////////////////////////

#ifndef RM_CCA
#define RM_CCA

#include <Eigen/Core>
#include <cstddef>
#include <vector>

namespace cc_affordance_planner
{

/** @brief Strict-priority CCA correction with base stationarity in the task null space. */
struct NullSpaceHierarchyStep
{
    Eigen::VectorXd delta;
    Eigen::VectorXd task_residual;
    Eigen::VectorXd base_delta;
};

/** @brief Bound-aware hierarchy result after feasible-variable redistribution. */
struct FeasibleNullSpaceStep
{
    Eigen::VectorXd delta;
    Eigen::VectorXd task_residual;
    Eigen::VectorXd base_delta;
    size_t active_arm_dof_count = 0;
    std::vector<size_t> frozen_variable_indices;
};

/**
 * @brief Computes a Moore-Penrose pseudoinverse with explicit relative singular-value truncation.
 *
 * Singular values sigma <= relative_tolerance * max(1, sigma_max) are treated as unavailable mobility directions.
 * The absolute floor prevents numerical noise in a theoretically zero projected matrix from becoming false mobility.
 */
Eigen::MatrixXd pseudo_inverse_svd(const Eigen::MatrixXd &matrix, double relative_tolerance);

/** @brief Selects matrix columns in the explicit order given by indices. */
Eigen::MatrixXd select_columns(const Eigen::MatrixXd &matrix, const std::vector<size_t> &indices);

/** @brief Adds scale * update into the explicitly indexed entries of target. */
void scatter_update(Eigen::VectorXd &target, const std::vector<size_t> &indices, const Eigen::VectorXd &update,
                    double scale = 1.0);

/**
 * @brief Computes Jc = -Ns^+ (Np + rho * theta_pdot^+) using truncated SVD.
 *
 * Jc maps a primary differential to the corresponding secondary differential in the current Newton iteration.
 */
Eigen::MatrixXd compute_closed_chain_mapping(const Eigen::MatrixXd &Np, const Eigen::MatrixXd &Ns,
                                              const Eigen::VectorXd &rho, const Eigen::VectorXd &theta_pdot,
                                              double svd_relative_tolerance);

/**
 * @brief Solves one strict-priority feasible-variable correction.
 *
 * The CCA task J * delta = error is primary. Base stationarity is secondary and is imposed only in the primary-task
 * null space:
 *
 * delta = J^+ error - N (B N)^+ B J^+ error, N = I - J^+ J.
 *
 * reserve_column_indices identify the columns selected by B.
 */
NullSpaceHierarchyStep compute_base_stationary_step(const Eigen::MatrixXd &jacobian,
                                                    const Eigen::VectorXd &error,
                                                    const std::vector<size_t> &reserve_column_indices,
                                                    double svd_relative_tolerance);

/**
 * @brief Applies the same strict hierarchy with a direction-aware bound active set.
 *
 * A correction is clipped to the first boundary, the corresponding variables are frozen only for the current solve,
 * and the unsatisfied primary correction is re-solved with the remaining variables. A subsequent Newton iteration
 * starts with every variable feasible again, so a joint at its upper bound can immediately reactivate for a negative
 * correction (and conversely at a lower bound).
 */
FeasibleNullSpaceStep compute_feasible_nullspace_step(
    const Eigen::MatrixXd &jacobian, const Eigen::VectorXd &error, const Eigen::VectorXd &state,
    const Eigen::VectorXd &lower_limits, const Eigen::VectorXd &upper_limits,
    const std::vector<size_t> &arm_column_indices, const std::vector<size_t> &reserve_column_indices,
    double svd_relative_tolerance, double bound_tolerance);

/** @brief Computes the current six-dimensional closed-chain FK error. */
Eigen::VectorXd compute_closure_error(const Eigen::MatrixXd &slist, const Eigen::VectorXd &theta_p,
                                      const Eigen::VectorXd &theta_s);

} // namespace cc_affordance_planner

#endif // RM_CCA
