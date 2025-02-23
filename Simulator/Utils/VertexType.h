/**
 * @file VertexType.h
 *
 * @ingroup Simulator/Utils
 * 
 * @brief Enum class of vertex types 
 */

// NETWORK MODEL VARIABLES NMV-BEGIN {
// Vertex types.
// NEURO:
//	INH - Inhibitory neuron
//	EXC - Excitory neuron
// NG911:
// CALR: Caller radii
// PSAP: PSAP nodes
// EMS, FIRE, LAW: Responder nodes

#ifndef VERTEX_TYPE_H
#define VERTEX_TYPE_H

#include <ostream>

enum class vertexType {
   // Neuro
   INH = 1,
   EXC = 2,
   // NG911
   CALR = 3,
   PSAP = 4,
   EMS = 5,
   FIRE = 6,
   LAW = 7,
   // UNDEF
   VTYPE_UNDEF = 0
};

// Custom streaming operator<< for the enum class vertexType
inline std::ostream &operator<<(std::ostream &os, vertexType vT)
{
   os << static_cast<int>(vT);
   return os;
}

#endif   // VERTEX_TYPE_H