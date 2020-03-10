/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
#ifndef ECL_MPI_SERIALIZER_HH
#define ECL_MPI_SERIALIZER_HH

#include <opm/simulators/utils/ParallelRestart.hpp>

namespace Opm {

class EclMpiSerializer {
public:
    enum class Operation {
        PACKSIZE,
        PACK,
        UNPACK
    };

    explicit EclMpiSerializer(Dune::CollectiveCommunication<Dune::MPIHelper::MPICommunicator> comm) :
        m_comm(comm)
    {}

    template<class T>
    void operator()(const T& data)
    {
        if (m_op == Operation::PACKSIZE)
            m_packSize += Mpi::packSize(data, m_comm);
        else if (m_op == Operation::PACK)
            Mpi::pack(data, m_buffer, m_position, m_comm);
        else if (m_op == Operation::UNPACK)
            Mpi::unpack(const_cast<T&>(data), m_buffer, m_position, m_comm);
    }

    template<class T>
    void pack(T& data)
    {
        m_op = Operation::PACKSIZE;
        m_packSize = 0;
        data.serializeOp(*this);
        m_position = 0;
        m_buffer.resize(m_packSize);
        m_op = Operation::PACK;
        data.serializeOp(*this);
    }

    template<class T>
    void unpack(T& data)
    {
        m_position = 0;
        m_op = Operation::UNPACK;
        data.serializeOp(*this);
    }

    template<class T>
    void broadcast(T& data)
    {
        if (m_comm.size() == 1)
            return;

#if HAVE_MPI
        if (m_comm.rank() == 0) {
            pack(data);
            m_comm.broadcast(&m_position, 1, 0);
            m_comm.broadcast(m_buffer.data(), m_position, 0);
        } else {
            m_comm.broadcast(&m_packSize, 1, 0);
            m_buffer.resize(m_packSize);
            m_comm.broadcast(m_buffer.data(), m_packSize, 0);
            unpack(data);
        }
#endif
    }

    size_t position() const
    {
        return m_position;
    }

protected:
    Dune::CollectiveCommunication<Dune::MPIHelper::MPICommunicator> m_comm;

    Operation m_op = Operation::PACKSIZE;
    size_t m_packSize = 0;
    int m_position = 0;
    std::vector<char> m_buffer;
};

}

#endif
