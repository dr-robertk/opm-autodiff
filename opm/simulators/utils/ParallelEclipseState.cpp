/*
  Copyright 2019 Equinor AS.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <config.h>

#include "ParallelEclipseState.hpp"
#include "ParallelRestart.hpp"
#include <ebos/eclmpiserializer.hh>

namespace Opm {


ParallelFieldPropsManager::ParallelFieldPropsManager(FieldPropsManager& manager)
    : m_manager(manager)
    , m_comm(Dune::MPIHelper::getCollectiveCommunication())
{
}


std::vector<int> ParallelFieldPropsManager::actnum() const
{
    if (m_comm.rank() == 0)
        return m_manager.actnum();

    return{};
}


void ParallelFieldPropsManager::reset_actnum(const std::vector<int>& actnum)
{
    if (m_comm.rank() != 0)
        OPM_THROW(std::runtime_error, "reset_actnum should only be called on root process.");
    m_manager.reset_actnum(actnum);
}


std::vector<double> ParallelFieldPropsManager::porv(bool global) const
{
    std::vector<double> result;
    if (m_comm.rank() == 0)
        result = m_manager.porv(global);
    size_t size = result.size();
    m_comm.broadcast(&size, 1, 0);
    result.resize(size);
    m_comm.broadcast(result.data(), size, 0);
    return result;
}


const std::vector<int>& ParallelFieldPropsManager::get_int(const std::string& keyword) const
{
    auto it = m_intProps.find(keyword);
    if (it == m_intProps.end())
        OPM_THROW(std::runtime_error, "No integer property field: " + keyword);

    return it->second;
}

std::vector<int> ParallelFieldPropsManager::get_global_int(const std::string& keyword) const
{
    std::vector<int> result;
    if (m_comm.rank() == 0)
        result = m_manager.get_global_int(keyword);
    size_t size = result.size();
    m_comm.broadcast(&size, 1, 0);
    result.resize(size);
    m_comm.broadcast(result.data(), size, 0);

    return result;
}


const std::vector<double>& ParallelFieldPropsManager::get_double(const std::string& keyword) const
{
    auto it = m_doubleProps.find(keyword);
    if (it == m_doubleProps.end())
        OPM_THROW(std::runtime_error, "No double property field: " + keyword);

    return it->second;
}


std::vector<double> ParallelFieldPropsManager::get_global_double(const std::string& keyword) const
{
    std::vector<double> result;
    if (m_comm.rank() == 0)
        result = m_manager.get_global_double(keyword);
    size_t size = result.size();
    m_comm.broadcast(&size, 1, 0);
    result.resize(size);
    m_comm.broadcast(result.data(), size, 0);

    return result;
}


bool ParallelFieldPropsManager::has_int(const std::string& keyword) const
{
    auto it = m_intProps.find(keyword);
    return it != m_intProps.end();
}


bool ParallelFieldPropsManager::has_double(const std::string& keyword) const
{
    auto it = m_doubleProps.find(keyword);
    return it != m_doubleProps.end();
}


ParallelEclipseState::ParallelEclipseState()
    : m_fieldProps(field_props)
{
}


ParallelEclipseState::ParallelEclipseState(const Deck& deck)
    : EclipseState(deck)
    , m_fieldProps(field_props)
{
}


#if HAVE_MPI
void ParallelEclipseState::serializeOp(EclMpiSerializer& serializer)
{
    serializer(m_tables);
    serializer(m_runspec);
    serializer(m_eclipseConfig);
    serializer(m_deckUnitSystem);
    serializer(m_inputNnc);
    serializer(m_inputEditNnc);
    serializer(m_gridDims);
    serializer(m_simulationConfig);
    serializer(m_transMult);
    serializer(m_faults);
    serializer(m_title);
    serializer(aquifer_config);
}
#endif


const FieldPropsManager& ParallelEclipseState::fieldProps() const
{
    if (!m_parProps && Dune::MPIHelper::getCollectiveCommunication().rank() != 0)
        OPM_THROW(std::runtime_error, "Attempt to access field properties on no-root process before switch to parallel properties");

    if (!m_parProps || Dune::MPIHelper::getCollectiveCommunication().size() == 1)
        return this->EclipseState::fieldProps();

    return m_fieldProps;
}


const FieldPropsManager& ParallelEclipseState::globalFieldProps() const
{
    if (Dune::MPIHelper::getCollectiveCommunication().rank() != 0)
        OPM_THROW(std::runtime_error, "Attempt to access global field properties on non-root process");
    return this->EclipseState::globalFieldProps();
}


const EclipseGrid& ParallelEclipseState::getInputGrid() const
{
    if (Dune::MPIHelper::getCollectiveCommunication().rank() != 0)
        OPM_THROW(std::runtime_error, "Attempt to access eclipse grid on non-root process");
    return this->EclipseState::getInputGrid();
}


void ParallelEclipseState::switchToGlobalProps()
{
    m_parProps = false;
}


void ParallelEclipseState::switchToDistributedProps()
{
    const auto& comm = Dune::MPIHelper::getCollectiveCommunication();
    if (comm.size() == 1) // No need for the parallel frontend
        return;

    m_parProps = true;
}

} // end namespace Opm
