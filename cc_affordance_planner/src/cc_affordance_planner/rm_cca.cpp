#include <cc_affordance_planner/rm_cca.hpp>

#include <Eigen/SVD>
#include <affordance_util/affordance_util.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace cc_affordance_planner
{
namespace
{
void validate_indices(const size_t upper_bound, const std::vector<size_t> &indices)
{
    std::unordered_set<size_t> unique;
    for (const size_t index : indices)
    {
        if (index >= upper_bound)
        {
            throw std::invalid_argument("RM-CCA index is out of range.");
        }
        if (!unique.insert(index).second)
        {
            throw std::invalid_argument("RM-CCA index vectors cannot contain duplicates.");
        }
    }
}

double feasible_alpha(const double state, const double delta, const double lower, const double upper)
{
    if (delta > 0.0)
    {
        return (upper - state) / delta;
    }
    if (delta < 0.0)
    {
        return (lower - state) / delta;
    }
    return std::numeric_limits<double>::infinity();
}

bool contains(const std::unordered_set<size_t> &indices, const size_t index)
{
    return indices.find(index) != indices.end();
}
} // namespace

Eigen::MatrixXd pseudo_inverse_svd(const Eigen::MatrixXd &matrix, const double relative_tolerance)
{
    if (!(relative_tolerance > 0.0) || !std::isfinite(relative_tolerance))
    {
        throw std::invalid_argument("SVD relative tolerance must be finite and positive.");
    }
    if (matrix.rows() == 0 || matrix.cols() == 0)
    {
        return Eigen::MatrixXd::Zero(matrix.cols(), matrix.rows());
    }
    if (!matrix.allFinite())
    {
        throw std::invalid_argument("Cannot compute an SVD pseudoinverse of a non-finite matrix.");
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd singular_values = svd.singularValues();
    Eigen::VectorXd inverse_values = Eigen::VectorXd::Zero(singular_values.size());
    if (singular_values.size() > 0)
    {
        // The absolute floor is essential for projected matrices such as B*N. When the usable null space is empty,
        // B*N is theoretically zero but floating-point projection leaves O(epsilon) entries; a purely relative
        // cutoff would normalize that numerical noise and invert it into an enormous false mobility direction.
        const double cutoff = relative_tolerance * std::max(1.0, singular_values(0));
        for (Eigen::Index i = 0; i < singular_values.size(); ++i)
        {
            if (singular_values(i) > cutoff)
            {
                inverse_values(i) = 1.0 / singular_values(i);
            }
        }
    }
    return svd.matrixV() * inverse_values.asDiagonal() * svd.matrixU().transpose();
}

Eigen::MatrixXd select_columns(const Eigen::MatrixXd &matrix, const std::vector<size_t> &indices)
{
    validate_indices(static_cast<size_t>(matrix.cols()), indices);
    Eigen::MatrixXd selected(matrix.rows(), static_cast<Eigen::Index>(indices.size()));
    for (size_t i = 0; i < indices.size(); ++i)
    {
        selected.col(static_cast<Eigen::Index>(i)) = matrix.col(static_cast<Eigen::Index>(indices[i]));
    }
    return selected;
}

void scatter_update(Eigen::VectorXd &target, const std::vector<size_t> &indices, const Eigen::VectorXd &update,
                    const double scale)
{
    validate_indices(static_cast<size_t>(target.size()), indices);
    if (update.size() != static_cast<Eigen::Index>(indices.size()))
    {
        throw std::invalid_argument("Scatter update size must match the index vector size.");
    }
    if (!std::isfinite(scale))
    {
        throw std::invalid_argument("Scatter update scale must be finite.");
    }
    for (size_t i = 0; i < indices.size(); ++i)
    {
        target(static_cast<Eigen::Index>(indices[i])) += scale * update(static_cast<Eigen::Index>(i));
    }
}

Eigen::MatrixXd compute_closed_chain_mapping(const Eigen::MatrixXd &Np, const Eigen::MatrixXd &Ns,
                                              const Eigen::VectorXd &rho, const Eigen::VectorXd &theta_pdot,
                                              const double svd_relative_tolerance)
{
    if (Np.rows() != Ns.rows() || rho.size() != Np.rows() || theta_pdot.size() != Np.cols())
    {
        throw std::invalid_argument("Closed-chain mapping matrix dimensions are inconsistent.");
    }
    const Eigen::MatrixXd theta_pdot_matrix = theta_pdot;
    return -pseudo_inverse_svd(Ns, svd_relative_tolerance) *
           (Np + rho * pseudo_inverse_svd(theta_pdot_matrix, svd_relative_tolerance));
}

NullSpaceHierarchyStep compute_base_stationary_step(const Eigen::MatrixXd &jacobian,
                                                    const Eigen::VectorXd &error,
                                                    const std::vector<size_t> &reserve_column_indices,
                                                    const double svd_relative_tolerance)
{
    if (jacobian.rows() != error.size() || !jacobian.allFinite() || !error.allFinite())
    {
        throw std::invalid_argument("Hierarchy Jacobian and error dimensions or values are invalid.");
    }
    validate_indices(static_cast<size_t>(jacobian.cols()), reserve_column_indices);

    NullSpaceHierarchyStep result;
    const Eigen::MatrixXd pseudoinverse = pseudo_inverse_svd(jacobian, svd_relative_tolerance);
    const Eigen::VectorXd whole_body_delta = pseudoinverse * error;
    const Eigen::MatrixXd nullspace =
        Eigen::MatrixXd::Identity(jacobian.cols(), jacobian.cols()) - pseudoinverse * jacobian;

    result.delta = whole_body_delta;
    if (!reserve_column_indices.empty())
    {
        Eigen::MatrixXd base_selector =
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(reserve_column_indices.size()), jacobian.cols());
        for (size_t i = 0; i < reserve_column_indices.size(); ++i)
        {
            base_selector(static_cast<Eigen::Index>(i),
                          static_cast<Eigen::Index>(reserve_column_indices[i])) = 1.0;
        }
        const Eigen::MatrixXd base_nullspace = base_selector * nullspace;
        result.delta -= nullspace * pseudo_inverse_svd(base_nullspace, svd_relative_tolerance) *
                        base_selector * whole_body_delta;
        result.base_delta = base_selector * result.delta;
    }
    else
    {
        result.base_delta = Eigen::VectorXd::Zero(0);
    }
    result.task_residual = error - jacobian * result.delta;
    return result;
}

FeasibleNullSpaceStep compute_feasible_nullspace_step(
    const Eigen::MatrixXd &jacobian, const Eigen::VectorXd &error, const Eigen::VectorXd &state,
    const Eigen::VectorXd &lower_limits, const Eigen::VectorXd &upper_limits,
    const std::vector<size_t> &arm_column_indices, const std::vector<size_t> &reserve_column_indices,
    const double svd_relative_tolerance, const double bound_tolerance)
{
    const Eigen::Index variable_count = jacobian.cols();
    if (jacobian.rows() != error.size() || state.size() != variable_count || lower_limits.size() != variable_count ||
        upper_limits.size() != variable_count || !jacobian.allFinite() || !error.allFinite() || !state.allFinite() ||
        bound_tolerance < 0.0 || !std::isfinite(bound_tolerance))
    {
        throw std::invalid_argument("Feasible hierarchy dimensions, values, or tolerance are invalid.");
    }
    validate_indices(static_cast<size_t>(variable_count), arm_column_indices);
    validate_indices(static_cast<size_t>(variable_count), reserve_column_indices);
    std::unordered_set<size_t> arm_set(arm_column_indices.begin(), arm_column_indices.end());
    const std::unordered_set<size_t> reserve_set(reserve_column_indices.begin(), reserve_column_indices.end());
    for (const size_t index : reserve_column_indices)
    {
        if (contains(arm_set, index))
        {
            throw std::invalid_argument("Arm and reserve hierarchy columns cannot overlap.");
        }
    }
    for (Eigen::Index i = 0; i < variable_count; ++i)
    {
        if (std::isnan(lower_limits(i)) || std::isnan(upper_limits(i)) || lower_limits(i) > upper_limits(i) ||
            state(i) < lower_limits(i) - bound_tolerance || state(i) > upper_limits(i) + bound_tolerance)
        {
            throw std::invalid_argument("Hierarchy state is outside an invalid bound interval.");
        }
    }

    FeasibleNullSpaceStep result;
    result.delta = Eigen::VectorXd::Zero(variable_count);
    result.active_arm_dof_count = arm_column_indices.size();
    std::vector<size_t> active_indices;
    active_indices.reserve(static_cast<size_t>(variable_count));
    for (Eigen::Index i = 0; i < variable_count; ++i)
    {
        if (upper_limits(i) - lower_limits(i) <= bound_tolerance)
        {
            const size_t fixed_index = static_cast<size_t>(i);
            result.frozen_variable_indices.push_back(fixed_index);
            if (arm_set.erase(fixed_index) > 0)
            {
                --result.active_arm_dof_count;
            }
        }
        else
        {
            active_indices.push_back(static_cast<size_t>(i));
        }
    }

    const double solve_tolerance =
        std::numeric_limits<double>::epsilon() * 100.0 * std::max(1.0, error.norm());
    for (Eigen::Index active_set_iteration = 0; active_set_iteration <= variable_count; ++active_set_iteration)
    {
        const Eigen::VectorXd remaining_error = error - jacobian * result.delta;
        if (remaining_error.norm() <= solve_tolerance || active_indices.empty())
        {
            break;
        }

        const Eigen::MatrixXd feasible_jacobian = select_columns(jacobian, active_indices);
        std::vector<size_t> local_reserve_indices;
        for (size_t local = 0; local < active_indices.size(); ++local)
        {
            if (contains(reserve_set, active_indices[local]))
            {
                local_reserve_indices.push_back(local);
            }
        }
        const NullSpaceHierarchyStep local_step = compute_base_stationary_step(
            feasible_jacobian, remaining_error, local_reserve_indices, svd_relative_tolerance);

        std::vector<double> candidate_alpha(active_indices.size(), std::numeric_limits<double>::infinity());
        double alpha = 1.0;
        for (size_t local = 0; local < active_indices.size(); ++local)
        {
            const Eigen::Index global = static_cast<Eigen::Index>(active_indices[local]);
            const double current = state(global) + result.delta(global);
            candidate_alpha[local] = feasible_alpha(current, local_step.delta(static_cast<Eigen::Index>(local)),
                                                    lower_limits(global), upper_limits(global));
            alpha = std::min(alpha, candidate_alpha[local]);
        }
        alpha = std::clamp(alpha, 0.0, 1.0);
        scatter_update(result.delta, active_indices, local_step.delta, alpha);
        if (alpha >= 1.0 - bound_tolerance)
        {
            break;
        }

        std::vector<size_t> newly_frozen;
        for (size_t local = 0; local < active_indices.size(); ++local)
        {
            if (candidate_alpha[local] <= alpha + bound_tolerance)
            {
                newly_frozen.push_back(active_indices[local]);
            }
        }
        if (newly_frozen.empty())
        {
            throw std::runtime_error("Bound active set clipped a correction without identifying a boundary.");
        }
        for (const size_t frozen : newly_frozen)
        {
            result.frozen_variable_indices.push_back(frozen);
            if (arm_set.erase(frozen) > 0)
            {
                --result.active_arm_dof_count;
            }
        }
        active_indices.erase(
            std::remove_if(active_indices.begin(), active_indices.end(), [&](const size_t index) {
                return std::find(newly_frozen.begin(), newly_frozen.end(), index) != newly_frozen.end();
            }),
            active_indices.end());
    }

    result.task_residual = error - jacobian * result.delta;
    result.base_delta.resize(static_cast<Eigen::Index>(reserve_column_indices.size()));
    for (size_t i = 0; i < reserve_column_indices.size(); ++i)
    {
        result.base_delta(static_cast<Eigen::Index>(i)) =
            result.delta(static_cast<Eigen::Index>(reserve_column_indices[i]));
    }
    return result;
}

Eigen::VectorXd compute_closure_error(const Eigen::MatrixXd &slist, const Eigen::VectorXd &theta_p,
                                      const Eigen::VectorXd &theta_s)
{
    if (slist.rows() != 6 || slist.cols() != theta_p.size() + theta_s.size())
    {
        throw std::invalid_argument("Closure error dimensions are inconsistent.");
    }
    Eigen::VectorXd thetalist(theta_p.size() + theta_s.size());
    thetalist << theta_p, theta_s;
    const Eigen::Matrix4d transform =
        affordance_util::FKinSpace(Eigen::Matrix4d::Identity(), slist, thetalist);
    return affordance_util::Adjoint(transform) *
           affordance_util::se3ToVec(affordance_util::MatrixLog6(affordance_util::TransInv(transform)));
}

} // namespace cc_affordance_planner
